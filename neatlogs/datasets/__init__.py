"""Capture instrumented local root executions into durable dataset jobs."""

from __future__ import annotations

import asyncio
import threading
import warnings
from uuid import UUID

from ._api import CaptureHandle, DatasetCaptureError, encode_manifest
from ._recording import Recording, current_recording, pipeline

__all__ = ["capture", "CaptureHandle", "DatasetCaptureError"]


def _warn_cleanup(message):
    try:
        warnings.warn(message, RuntimeWarning, stacklevel=3)
    except RuntimeWarning:
        # A caller's warnings-as-errors policy must not replace its agent exception.
        pass


class _Capture:
    def __init__(self, destination, handle):
        self.handle = handle
        self.destination = destination
        self.recording = None
        self._entered = False
        self._entry_lock = threading.Lock()

    def _prepare(self):
        if current_recording.get() is not None:
            raise DatasetCaptureError("Nested dataset capture is not supported")
        with self._entry_lock:
            if self._entered:
                raise DatasetCaptureError("A dataset capture context cannot be reused")
            processor, self._flush = pipeline()
            self.handle._register(self.destination)
            self.recording = Recording(processor)
            self._entered = True

    def __enter__(self):
        self._prepare()
        self._context_token = current_recording.set(self.recording)
        return self.handle

    async def __aenter__(self):
        task = asyncio.create_task(asyncio.to_thread(self._prepare))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(task)
                await asyncio.to_thread(self.handle._abort, "cancelled")
            except Exception as error:
                self.handle.error = error
                _warn_cleanup("Cancelled capture cleanup failed; inspect the capture handle")
            raise
        self._context_token = current_recording.set(self.recording)
        return self.handle

    def _finish(self, original_error):
        reason = "incomplete_execution"
        try:
            manifest = self.recording.freeze()
            encode_manifest(manifest)
            reason = "export_failed"
            if manifest["traces"]:
                try:
                    self._flush()
                except Exception:
                    # A shared pipeline may fail unrelated work; only our receipts decide success.
                    pass
            try:
                exported = self.recording.exported_manifest()
            except ValueError as error:
                raise DatasetCaptureError(
                    "Trace export lost data; dataset capture was aborted"
                ) from error
            if isinstance(original_error, (asyncio.CancelledError, KeyboardInterrupt)):
                reason = "cancelled"
                self.handle._abort(reason)
                return
            self.handle._manifest = encode_manifest(exported)
            # An ambiguous completion response must be retried, never countermanded by abort.
            reason = None
            self.handle.retry_completion()
        except Exception as error:
            self.handle.error = error
            if reason is not None:
                try:
                    self.handle._abort(reason)
                except Exception:
                    _warn_cleanup(
                        "Dataset capture abort could not be confirmed; inspect the capture handle"
                    )
            if original_error is None:
                raise DatasetCaptureError(str(error)) from error
            _warn_cleanup(
                "Dataset capture failed while preserving the agent exception; inspect handle.error"
            )
        finally:
            self.recording.closed = True
            self.recording.processor.release(self.recording)

    def __exit__(self, exc_type, exc, tb):
        current_recording.reset(self._context_token)
        self._finish(exc)
        return False

    async def __aexit__(self, exc_type, exc, tb):
        current_recording.reset(self._context_token)
        task = asyncio.create_task(asyncio.to_thread(self._finish, exc))
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            try:
                await asyncio.shield(task)
            except Exception as error:
                self.handle.error = error
                _warn_cleanup("Cancelled capture cleanup failed; inspect the capture handle")
            raise
        return False


def capture(
    name: str | None = None,
    *,
    dataset_id: str | None = None,
    project_id: str,
    token: str,
    base_url: str,
    environment: str | None = None,
    create_if_missing: bool = False,
    allow_production: bool = False,
    description: str | None = None,
):
    """Register, run, flush and queue dataset capture using ``with`` or ``async with``.

    ``token`` is a public API user/service-account credential, never an ingestion key.
    Only root traces started in this scope and their locally recorded spans are captured.
    Await all work inside the block; distributed descendants are not implied complete.
    The environment describes the agent workload, independently of the backend hostname.
    """
    if environment not in {"local", "ci", "production"}:
        raise ValueError("Set environment explicitly to 'local', 'ci', or 'production'")
    if not isinstance(create_if_missing, bool) or not isinstance(allow_production, bool):
        raise ValueError("create_if_missing and allow_production must be booleans")
    if environment == "production" and not allow_production:
        raise ValueError("Production dataset capture requires allow_production=True")
    if (name is None) == (dataset_id is None):
        raise ValueError("Provide exactly one of name or dataset_id")
    if dataset_id is not None:
        if create_if_missing or description is not None:
            raise ValueError("create_if_missing and description apply only to named destinations")
        destination = {"kind": "existing", "datasetId": str(UUID(dataset_id))}
    else:
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 255:
            raise ValueError("name must contain 1–255 characters")
        destination = {"kind": "name", "name": name.strip(), "createIfMissing": create_if_missing}
        if description is not None:
            if not isinstance(description, str) or len(description) > 2000:
                raise ValueError("description must be at most 2000 characters")
            destination["description"] = description
    return _Capture(
        destination, CaptureHandle(project_id=project_id, token=token, base_url=base_url)
    )
