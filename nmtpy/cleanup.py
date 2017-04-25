# -*- coding: utf-8 -*-
import os
import sys
import signal
import atexit
import traceback

temp_files = set()
subprocesses = set()

def register_tmp_file(f):
    """Add new temp file to global set."""
    temp_files.add(f)

def register_proc(pid):
    """Add new process to global set."""
    subprocesses.add(pid)

def unregister_proc(pid):
    """Remove given PID from global set."""
    subprocesses.remove(pid)

def cleanup():
    """Cleanup registered temp files and kill PIDs."""
    for f in temp_files:
        try:
            os.unlink(f)
        except:
            pass

    for p in subprocesses:
        try:
            os.kill(p, signal.SIGTERM)
        except:
            pass

def signal_handler(signum, frame):
    """Let Python call this when SIGINT or SIGTERM caught."""
    cleanup()
    sys.exit(0)

def register_exception_handler(logger, quit_on_exception=False):
    """Setup exception handler."""

    def exception_handler(exctype, value, tb):
        """Let Python call this when an exception is uncaught."""
        logger.info(''.join(traceback.format_exception(exctype, value, tb)))

    def exception_handler_quits(exctype, value, tb):
        """Let Python call this when an exception is uncaught."""
        logger.info(''.join(traceback.format_exception(exctype, value, tb)))
        sys.exit(1)

    if quit_on_exception:
        sys.excepthook = exception_handler_quits
    else:
        sys.excepthook = exception_handler

def register_handler(logger, _atexit=True, _signals=True, exception_quits=False):
    """Register atexit and signal handlers."""
    if _atexit:
        # Register exit handler
        atexit.register(cleanup)

    if _signals:
        # Register SIGINT and SIGTERM
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    register_exception_handler(logger, exception_quits)
