"""Task-file entry point used by ``neatlogs evals run``."""

import contextlib
import importlib.util
import json
import os
import sys
from pathlib import Path

from ..datasets._api import DatasetError
from ..replay.api import LocalRunApiError
from . import Dataset, run_eval


def main():
    try:
        path = Path(sys.argv[1]).resolve(strict=True)
        options = json.loads(os.environ["NEATLOGS_EVAL_OPTIONS"])
        spec = importlib.util.spec_from_file_location("neatlogs_eval_task", path)
        if spec is None or spec.loader is None:
            raise ValueError("Use a Python task file exporting task(input, context)")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        with contextlib.redirect_stdout(sys.stderr):
            spec.loader.exec_module(module)
            task = getattr(module, "task", None)
            if not callable(task):
                raise ValueError("Task file must export task(input, context)")
            selector = options.pop("dataset")
            dataset = Dataset(
                selector["name"], version_id=selector.get("versionId"), tag=selector.get("tag")
            )
            result = run_eval(dataset=dataset, task=task, **options)
        print(
            json.dumps(
                {
                    "runId": result.run_id,
                    "comparison": result.comparison,
                    "gate": result.gate,
                    "errors": result.errors,
                    "exitCode": result.exit_code,
                },
                allow_nan=False,
            )
        )
        return result.exit_code
    except (DatasetError, LocalRunApiError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        status = getattr(error, "status", getattr(error, "status_code", None))
        return 2 if status in {400, 422} else 3
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
