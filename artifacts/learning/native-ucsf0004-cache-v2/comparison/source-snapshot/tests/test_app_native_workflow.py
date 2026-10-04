"""Explicit native integration runner (not a displayless pytest test).

Run with ``python tests/test_app_native_workflow.py --case CASE --output DIR``.
It exercises actual Qt callbacks, VTK pixels, route selectors, persistence and
cancellation while an event-loop heartbeat measures responsiveness.
"""

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from resectionlab.app.desktop import MainWindow
from resectionlab.app.style import STYLE
from resectionlab.imaging import create_synthetic_case, load_case


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    app = QApplication([])
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    window.scene.Initialize()
    window.scene.show_orientation()
    state = {"phase": "loading", "ticks": [], "started": time.perf_counter(), "checks": {}}
    bundle = args.output / "roundtrip.ressectionlab"

    def finish(error=None):
        timer.stop()
        times = np.diff(state["ticks"]) * 1000
        report = {"passed": error is None, "error": error, "checks": state["checks"],
                  "elapsed_seconds": time.perf_counter() - state["started"],
                  "heartbeat_p95_ms": float(np.percentile(times, 95)) if len(times) else None,
                  "heartbeat_max_ms": float(max(times)) if len(times) else None,
                  "case_hash": window.case.semantic_hash if window.case else None}
        (args.output / "workflow.json").write_text(json.dumps(report, indent=2))
        app.exit(0 if error is None else 1)

    def tick():
        state["ticks"].append(time.perf_counter())
        try:
            if time.perf_counter() - state["started"] > 45:
                raise TimeoutError(f"Native workflow timed out in {state['phase']}")
            phase = state["phase"]
            if phase == "loading" and window.case is not None and window.current_job is None:
                state["case_hash"] = window.case.semantic_hash
                window.support_checkbox.setChecked(window.case.brain_mask is None)
                window.generate_routes()
                state["phase"] = "planning"
            elif phase == "planning" and window.routes and window.current_job is None:
                assert window.compare_a.count() >= 2
                window.compare_a.setCurrentIndex(0)
                window.compare_b.setCurrentIndex(2)
                assert len(window.scene.route_actors) >= 2
                state["checks"]["two_accessible_route_selectors"] = True
                state["comparison"] = window.compare_b.currentData()
                state["routes"] = len(window.routes)
                window.save_workspace_image(args.output / "two-routes.png")
                window.opacity.setValue(42)
                cursor = window.cursor.copy()
                cursor[2] += 2
                window.set_cursor(cursor)
                state["cursor"] = cursor
                window.save_to(bundle)
                state["phase"] = "saving"
            elif phase == "saving" and window.current_job is None and bundle.exists():
                window.load_with(lambda: load_case(bundle), bundle_path=bundle)
                state["phase"] = "reopening"
            elif phase == "reopening" and window.current_job is None:
                assert window.case.semantic_hash == state["case_hash"]
                assert len(window.routes) == state["routes"]
                assert window.compare_b.currentData() == state["comparison"]
                assert np.allclose(window.cursor, state["cursor"])
                assert window.opacity.value() == 42
                state["checks"]["case_routes_view_roundtrip"] = True
                def cancellable(cancel, progress):
                    for _ in range(100):
                        if cancel.wait(.02):
                            return "discarded"
                    return "unexpected completion"
                window.start_job(cancellable, lambda _: finish("Cancelled result was published"), "Testing cancellation")
                QTimer.singleShot(100, window.cancel_job)
                state["phase"] = "cancelling"
            elif phase == "cancelling" and window.current_job is None:
                assert window.case.semantic_hash == state["case_hash"]
                state["checks"]["cancellation_preserves_case"] = True
                pixels = window.scene.capture_rgb()
                assert np.ptp(pixels.astype(float)) > 80
                state["checks"]["real_vtk_framebuffer"] = True
                window.save_workspace_image(args.output / "reopened.png")
                finish()
        except Exception as error:
            finish(f"{type(error).__name__}: {error}")

    timer = QTimer()
    timer.timeout.connect(tick)
    timer.start(20)
    if args.case:
        window.load_with(lambda: load_case(args.case), bundle_path=args.case)
    else:
        window.load_with(create_synthetic_case)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
