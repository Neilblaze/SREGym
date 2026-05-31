"""
Problem: An incomplete rollout leaves a previous build serving behind a live Service.

This problem models a deployment-process failure on the Hotel Reservation app. The production `frontend` Deployment is 
healthy and Running/Ready, but a second, unmanaged workload is still running a previous container build and shares the 
`frontend` Service selector. Because a Kubernetes Service load-balances across every pod whose labels match its selector, 
the Service routes a fraction of requests onto the stale pods even though the production Deployment looks perfectly 
healthy. Deleting the stale pods or restarting the production Deployment does not help, because the stray workload 
reconciles its pods back on its own.

CC'ing https://github.com/SREGym/SREGym/issues/834, the real-world archetype is the 2012 Knight Capital 
incident: A deploy reached seven of eight servers, the eighth kept running a previous build, and a reused 
configuration flag reactivated dormant code on that one host. The healthy hosts and the stale host served 
the same entry point, so behavior depended on which host answered a given request.
"""

import copy
import json
import time

import yaml

from sregym.conductor.oracles.llm_as_a_judge.llm_as_a_judge_oracle import LLMAsAJudgeOracle
from sregym.conductor.oracles.version_skew_mitigation import VersionSkewMitigationOracle
from sregym.conductor.problems.base import Problem
from sregym.service.apps.hotel_reservation import HotelReservation
from sregym.service.kubectl import KubeCtl
from sregym.utils.decorators import mark_fault_injected


class IncompleteRolloutVersionSkew(Problem):
    """Run a previous build as a stray workload behind the live frontend Service."""

    FAULTY_SERVICE = "frontend"
    # Previous build of the Hotel Reservation monolith. It is a real, pullable image
    # that diverges from the current build (the same tag the correlated-image problems
    # treat as the broken version), so the stray pods serve a genuinely different version.
    STALE_IMAGE = "jackcuii/hotel-reservation:latest"
    # Fallback only; the live current image is captured at injection time.
    CURRENT_IMAGE_FALLBACK = "yinfangchen/hotel-reservation:latest"
    # Generic, production-plausible name. It does not reveal the benchmark or the fault
    # category; identifying that this workload pollutes the live Service is the task.
    STALE_WORKLOAD_NAME = "frontend-canary"
    # Distinguishing label so the stray workload's ReplicaSet owns only its own pods and
    # never contends with the production Deployment's ReplicaSet for pod adoption.
    TRACK_LABEL_KEY = "track"
    TRACK_LABEL_VALUE = "canary"
    POD_TEMPLATE_HASH_LABEL = "pod-template-hash"

    def __init__(self):
        self.app = HotelReservation()
        self.namespace = self.app.namespace
        self.faulty_service = self.FAULTY_SERVICE

        super().__init__(app=self.app, namespace=self.namespace)

        self.kubectl = KubeCtl()

        self.stale_image = self.STALE_IMAGE
        self.stale_workload = self.STALE_WORKLOAD_NAME
        # Overwritten with the live image during inject_fault; kept for the oracle and
        # for human-readable messages before injection has run.
        self.current_image = self.CURRENT_IMAGE_FALLBACK

        self.root_cause = self.build_structured_root_cause(
            component=(
                f"Deployment/{self.stale_workload} serving behind Service/{self.faulty_service}"
            ),
            namespace=self.namespace,
            description=(
                f"An incomplete rollout left a previous build of `{self.faulty_service}` running as a "
                f"separate, unmanaged workload (`Deployment/{self.stale_workload}`). That workload carries "
                f"the same labels that `Service/{self.faulty_service}` selects on, so the Service load-balances "
                f"a fraction of requests onto its pods, which run an older container image than the production "
                f"`Deployment/{self.faulty_service}`. The production Deployment itself is healthy and "
                "Running/Ready, which is why pod-level health checks look fine; the defect is the version skew "
                "behind a single Service, caused by the extra workload rather than by any unhealthy pod in the "
                "production Deployment. A complete diagnosis should identify that more than one workload backs "
                f"`Service/{self.faulty_service}` and that the extra workload serves a stale image. A valid "
                "mitigation removes that stray workload from the Service (delete it, scale it to zero, or relabel "
                "it out of the selector) so that every pod behind the Service is managed by the production "
                "Deployment and runs the current build. Deleting the stale pods or restarting the production "
                "Deployment is not sufficient: the stray workload reconciles its pods back."
            ),
        )
        self.diagnosis_oracle = LLMAsAJudgeOracle(problem=self, expected=self.root_cause)

        self.app.create_workload()

        self.mitigation_oracle = VersionSkewMitigationOracle(
            problem=self,
            service_name=self.faulty_service,
            deployment_name=self.faulty_service,
        )

    @mark_fault_injected
    def inject_fault(self):
        print("== Fault Injection ==")
        deployment = self._get_deployment_json(self.faulty_service)
        self.current_image = self._primary_container_image(deployment)
        if self.current_image == self.stale_image:
            raise RuntimeError(
                f"Deployment/{self.faulty_service} already runs {self.stale_image}; cannot model version skew "
                "against an identical image."
            )

        service_selector = self._service_selector(self.faulty_service)
        self._create_stale_workload(deployment, service_selector)
        self._wait_for_version_skew(service_selector)
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")

    @mark_fault_injected
    def recover_fault(self):
        print("== Fault Recovery ==")
        self.kubectl.exec_command(
            f"kubectl delete deployment {self.stale_workload} -n {self.namespace} --ignore-not-found"
        )
        self._wait_for_skew_cleared(self._service_selector(self.faulty_service))
        print(f"Service: {self.faulty_service} | Namespace: {self.namespace}\n")


    def _create_stale_workload(self, deployment: dict, service_selector: dict):
        template = deployment["spec"]["template"]

        template_labels = dict(template.get("metadata", {}).get("labels", {}))
        selector_labels = dict(deployment["spec"]["selector"]["matchLabels"])
        # pod-template-hash is owned by the Deployment controller for its own ReplicaSets;
        # copying it would let two controllers fight over the same pods.
        template_labels.pop(self.POD_TEMPLATE_HASH_LABEL, None)
        selector_labels.pop(self.POD_TEMPLATE_HASH_LABEL, None)

        stale_template_labels = {**template_labels, self.TRACK_LABEL_KEY: self.TRACK_LABEL_VALUE}
        stale_selector_labels = {**selector_labels, self.TRACK_LABEL_KEY: self.TRACK_LABEL_VALUE}

        # Copy the live pod spec verbatim and change only the image, so the single difference
        # between the production pods and the stray pods is the container build.
        pod_spec = copy.deepcopy(template["spec"])
        for container in pod_spec.get("containers", []):
            container["image"] = self.stale_image

        replicas = deployment["spec"].get("replicas") or 1

        stale_deployment = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {
                "name": self.stale_workload,
                "namespace": self.namespace,
                "labels": stale_template_labels,
            },
            "spec": {
                "replicas": replicas,
                "selector": {"matchLabels": stale_selector_labels},
                "template": {
                    "metadata": {"labels": stale_template_labels},
                    "spec": pod_spec,
                },
            },
        }

        manifest_path = f"/tmp/{self.stale_workload}.yaml"
        with open(manifest_path, "w") as file:
            yaml.safe_dump(stale_deployment, file)

        apply_out = self.kubectl.exec_command(f"kubectl apply -f {manifest_path} -n {self.namespace}")
        print(f"Created stray workload Deployment/{self.stale_workload}: {apply_out.strip()}")

        missing = [key for key in service_selector if stale_template_labels.get(key) != service_selector[key]]
        if missing:
            raise RuntimeError(
                f"Stray workload labels do not satisfy Service/{self.faulty_service} selector for keys {missing}; "
                "the version skew would not reach the Service."
            )

    def _wait_for_version_skew(self, service_selector: dict, timeout: int = 180):
        """Block until Service/<faulty_service> selects at least one pod on the stale image."""
        selector_str = self._selector_to_str(service_selector)
        deadline = time.monotonic() + timeout
        last_seen = "no pods on the stale image yet"

        while time.monotonic() < deadline:
            pods = self._get_json(
                f"kubectl get pods -n {self.namespace} -l '{selector_str}' -o json",
                f"pods for Service/{self.faulty_service}",
            )
            stale_pods = [pod for pod in pods.get("items", []) if self._pod_uses_image(pod, self.stale_image)]
            if stale_pods:
                names = ", ".join(pod.get("metadata", {}).get("name", "<unknown>") for pod in stale_pods)
                print(f"Service/{self.faulty_service} now load-balances onto stale pods: {names}")
                return
            last_seen = f"{len(pods.get('items', []))} pod(s) selected, none on {self.stale_image}"
            time.sleep(5)

        raise RuntimeError(
            f"Service/{self.faulty_service} did not pick up a stale-image backend within {timeout}s. "
            f"Last observation: {last_seen}"
        )
    

    def _wait_for_skew_cleared(self, service_selector: dict, timeout: int = 180):
        selector_str = self._selector_to_str(service_selector)
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            pods = self._get_json(
                f"kubectl get pods -n {self.namespace} -l '{selector_str}' -o json",
                f"pods for Service/{self.faulty_service}",
            )
            stale_pods = [pod for pod in pods.get("items", []) if self._pod_uses_image(pod, self.stale_image)]
            if not stale_pods:
                return
            time.sleep(5)

        print(
            f"Warning: stale-image pods still behind Service/{self.faulty_service} after {timeout}s; "
            "recovery may be incomplete."
        )


    def _get_deployment_json(self, service: str) -> dict:
        return self._get_json(
            f"kubectl get deployment {service} -n {self.namespace} -o json", f"Deployment/{service}"
        )

    def _service_selector(self, service: str) -> dict:
        svc = self._get_json(f"kubectl get service {service} -n {self.namespace} -o json", f"Service/{service}")
        selector = svc.get("spec", {}).get("selector") or {}
        if not selector:
            raise RuntimeError(f"Service/{service} has no selector; cannot model Service-level version skew.")
        return selector

    @staticmethod
    def _primary_container_image(deployment: dict) -> str:
        containers = deployment.get("spec", {}).get("template", {}).get("spec", {}).get("containers", [])
        if not containers:
            raise RuntimeError("Deployment has no containers in its pod template.")
        return containers[0]["image"]

    @staticmethod
    def _pod_uses_image(pod: dict, image: str) -> bool:
        return any(container.get("image") == image for container in pod.get("spec", {}).get("containers", []))

    @staticmethod
    def _selector_to_str(selector: dict) -> str:
        return ",".join(f"{key}={value}" for key, value in sorted(selector.items()))

    def _get_json(self, command: str, resource_name: str) -> dict:
        raw = self.kubectl.exec_command(command).strip()
        if not raw:
            raise RuntimeError(f"kubectl returned no output for {resource_name}")
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Failed to parse kubectl JSON for {resource_name}: {exc}; output={raw[:500]!r}"
            ) from exc
        
