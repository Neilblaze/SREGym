"""
Mitigation oracle for Service-level version skew.

The default MitigationOracle only checks per-pod health, which is insufficient here, since the production Deployment stays 
Running/Ready while a separate, unmanaged workload serves a previous build behind the same Service. This oracle instead
verifies the invariant that actually defines the fix, i.e., every pod backing the target Service must be owned by the intended 
Deployment and must not run the previous build. That makes the canonical fix (remove the stray workload from the Service) 
pass, while the common reflex fixes that leave the stray workload in place do not:

* deleting the stale pods                               -> the stray workload reconciles them back
* restarting the production Deployment                  -> the stray workload is untouched
* re-tagging the stray workload to the current image    -> a second, unmanaged workload is still serving the production Service
"""

from __future__ import annotations

import json
import time
from json import JSONDecodeError

from sregym.conductor.oracles.base import Oracle

_DEFAULT_TIMEOUT_SECONDS = 90
_DEFAULT_POLL_INTERVAL_SECONDS = 5
_DEFAULT_CONSECUTIVE_HEALTHY_POLLS = 2


class VersionSkewMitigationOracle(Oracle):
    """Pass when the target Service is served only by the intended Deployment's current build"""

    importance = 1.0

    def __init__(
        self,
        problem,
        *,
        service_name: str,
        deployment_name: str,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        poll_interval_seconds: int = _DEFAULT_POLL_INTERVAL_SECONDS,
        consecutive_successes: int = _DEFAULT_CONSECUTIVE_HEALTHY_POLLS,
    ):
        super().__init__(problem)
        self.service_name = service_name
        self.deployment_name = deployment_name
        self.timeout_seconds = timeout_seconds
        self.poll_interval_seconds = poll_interval_seconds
        self.consecutive_successes = consecutive_successes


    def evaluate(self) -> dict:
        print("== Version Skew Mitigation Evaluation ==")

        # Require several consecutive healthy polls so a transient window (for example, the
        # stray pods still terminating) is not graded as a pass.
        consecutive_healthy = 0
        last_detail = "not evaluated"
        deadline = time.monotonic() + self.timeout_seconds

        while True:
            healthy, detail = self._evaluate_once()
            last_detail = detail

            if healthy:
                consecutive_healthy += 1
                print(f"Healthy poll {consecutive_healthy}/{self.consecutive_successes}: {detail}")
                if consecutive_healthy >= self.consecutive_successes:
                    return {"success": True, "details": detail}
            else:
                if consecutive_healthy:
                    print("Health regressed; resetting consecutive poll count")
                consecutive_healthy = 0
                print(f"Not healthy: {detail}")

            if time.monotonic() >= deadline:
                return {
                    "success": False,
                    "details": (
                        f"Timed out after {self.timeout_seconds}s waiting for "
                        f"{self.consecutive_successes} consecutive healthy polls. Last check: {last_detail}"
                    ),
                }

            time.sleep(self.poll_interval_seconds)


    def _evaluate_once(self) -> tuple[bool, str]:
        namespace = self.problem.namespace

        deployment, error = self._kubectl_json(
            f"kubectl get deployment {self.deployment_name} -n {namespace} -o json"
        )
        if error:
            return False, error

        ready, detail = self._deployment_ready(deployment)
        if not ready:
            return False, detail

        service, error = self._kubectl_json(f"kubectl get service {self.service_name} -n {namespace} -o json")
        if error:
            return False, error

        selector = service.get("spec", {}).get("selector") or {}
        if not selector:
            return False, f"Service/{self.service_name} has no selector"
        selector_str = ",".join(f"{key}={value}" for key, value in sorted(selector.items()))

        pods, error = self._kubectl_json(f"kubectl get pods -n {namespace} -l '{selector_str}' -o json")
        if error:
            return False, error

        items = pods.get("items", [])
        if not items:
            return False, f"Service/{self.service_name} selects no pods"

        owned_replicasets = {
            rs.metadata.name for rs in self.problem.kubectl.get_matching_replicasets(namespace, self.deployment_name)
        }
        stale_image = getattr(self.problem, "stale_image", None)

        for pod in items:
            pod_name = pod.get("metadata", {}).get("name", "<unknown>")

            if pod.get("metadata", {}).get("deletionTimestamp"):
                return False, f"Pod {pod_name} backing Service/{self.service_name} is terminating"

            phase = pod.get("status", {}).get("phase")
            if phase != "Running":
                return False, f"Pod {pod_name} backing Service/{self.service_name} is in phase {phase}"

            for container_status in pod.get("status", {}).get("containerStatuses", []):
                if not container_status.get("ready", False):
                    return False, (
                        f"Container {container_status.get('name')} in pod {pod_name} "
                        f"backing Service/{self.service_name} is not Ready"
                    )

            if stale_image and self._pod_uses_image(pod, stale_image):
                return False, (
                    f"Pod {pod_name} backing Service/{self.service_name} still runs the previous build "
                    f"{stale_image}"
                )

            owner_rs = self._owning_replicaset(pod)
            if owner_rs is None or owner_rs not in owned_replicasets:
                return False, (
                    f"Pod {pod_name} backing Service/{self.service_name} is not managed by "
                    f"Deployment/{self.deployment_name} (owning ReplicaSet={owner_rs!r}); a separate "
                    "workload is still serving this Service"
                )

        return True, (
            f"All pods backing Service/{self.service_name} are managed by "
            f"Deployment/{self.deployment_name} and run a single current build"
        )


    def _deployment_ready(self, deployment: dict) -> tuple[bool, str]:
        desired = deployment.get("spec", {}).get("replicas", 1)
        status = deployment.get("status", {})
        ready = status.get("readyReplicas", 0)
        updated = status.get("updatedReplicas", 0)
        unavailable = status.get("unavailableReplicas", 0)

        if desired < 1:
            return False, f"Deployment/{self.deployment_name} has desired replicas={desired}; expected at least 1"

        if ready < desired or updated < desired or unavailable:
            return False, (
                f"Deployment/{self.deployment_name} rollout not ready: "
                f"ready={ready}, updated={updated}, unavailable={unavailable}, desired={desired}"
            )

        return True, f"Deployment/{self.deployment_name} has {ready}/{desired} ready replicas"

    @staticmethod
    def _owning_replicaset(pod: dict) -> str | None:
        for owner in pod.get("metadata", {}).get("ownerReferences", []):
            if owner.get("kind") == "ReplicaSet":
                return owner.get("name")
        return None

    @staticmethod
    def _pod_uses_image(pod: dict, image: str) -> bool:
        return any(container.get("image") == image for container in pod.get("spec", {}).get("containers", []))

    def _kubectl_json(self, command: str) -> tuple[dict | None, str | None]:
        output = self.problem.kubectl.exec_command(command)
        stripped = output.strip()

        if not stripped:
            return None, f"Command returned no output: {command}"

        try:
            return json.loads(stripped), None
        except JSONDecodeError as exc:
            return None, f"Failed to parse JSON from `{command}`: {exc}; output={stripped[:500]!r}"
        
