import time
import requests
from config import N8N_API_KEY, N8N_API_BASE, N8N_BASE_URL, EXECUTION_POLL_INTERVAL, EXECUTION_TIMEOUT


class N8nAPIError(Exception):
    def __init__(self, message: str, status_code: int = 0, response_body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class N8nClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "X-N8N-API-KEY": N8N_API_KEY,
            "Content-Type": "application/json",
            "Accept": "application/json",
        })

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = f"{N8N_API_BASE}{path}"
        resp = self.session.request(method, url, **kwargs)
        if not resp.ok:
            raise N8nAPIError(
                f"n8n API error {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
                response_body=resp.text,
            )
        if resp.text.strip():
            return resp.json()
        return {}

    def create_workflow(self, workflow_data: dict) -> dict:
        """POST /api/v1/workflows — returns created workflow with id."""
        return self._request("POST", "/workflows", json=workflow_data)

    def activate_workflow(self, workflow_id: str) -> dict:
        """POST /api/v1/workflows/{id}/activate."""
        return self._request("POST", f"/workflows/{workflow_id}/activate")

    def delete_workflow(self, workflow_id: str) -> None:
        """DELETE /api/v1/workflows/{id}."""
        try:
            self._request("DELETE", f"/workflows/{workflow_id}")
        except N8nAPIError:
            pass  # Best-effort cleanup

    def get_executions(self, workflow_id: str, limit: int = 5) -> list:
        """GET /api/v1/executions?workflowId={id}&limit={n}."""
        result = self._request("GET", f"/executions?workflowId={workflow_id}&limit={limit}")
        return result.get("data", [])

    def get_execution(self, execution_id: str) -> dict:
        """GET /api/v1/executions/{id}."""
        return self._request("GET", f"/executions/{execution_id}?includeData=true")

    def trigger_webhook(self, webhook_path: str, payload: dict | None = None) -> dict | None:
        """Trigger a webhook node using its test URL."""
        # n8n webhook test URLs follow this pattern:
        # https://<instance>/webhook-test/<path>
        url = f"{N8N_BASE_URL}/webhook-test/{webhook_path}"
        try:
            resp = requests.post(url, json=payload or {"test": True}, timeout=30)
            if resp.ok and resp.text.strip():
                return resp.json()
            return {"status": resp.status_code, "body": resp.text}
        except requests.RequestException as e:
            return {"error": str(e)}

    def poll_for_execution(self, workflow_id: str, started_after: float) -> dict | None:
        """Poll executions until one finishes or timeout is reached."""
        deadline = time.time() + EXECUTION_TIMEOUT
        while time.time() < deadline:
            time.sleep(EXECUTION_POLL_INTERVAL)
            executions = self.get_executions(workflow_id, limit=5)
            for execution in executions:
                # Check if this execution started after we triggered the webhook
                exec_started = execution.get("startedAt", "")
                if execution.get("status") in ("success", "error", "crashed"):
                    return self.get_execution(execution["id"])
        return None
