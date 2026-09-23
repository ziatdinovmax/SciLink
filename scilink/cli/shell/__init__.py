"""SciLink's terminal shell — one chat surface for the four modes.

``scilink`` (meta), ``scilink analyze``, ``scilink plan`` and ``scilink
simulate`` all run the same shell over a mode adapter (``modes.py``). The
shell borrows the web backend's turn machinery — per-thread stdout capture
with a print-driven stop, a parked human-in-the-loop question, the
presenter's widget vocabulary — so the terminal and the browser say the
same things the same way (``scilink/ui/vocabulary.py``).

Layout:
  app.py        argument parsing, ``main(argv)``
  shell.py      the prompt loop, key bindings, banner
  turn.py       one chat turn: worker thread + capture + live rendering
  render.py     rich rendering of the narration and the answer
  channel.py    the HITL channel and its question widgets
  commands.py   slash commands (core set; modes add theirs)
  modes.py      what each mode contributes: flags, build, autonomy, extras
  bootstrap.py  credentials, session directory, consent, custom extras
  sessions.py   the resume picker
  headless.py   ``-p`` print mode over ``run_task``
"""

from .app import main

__all__ = ["main"]
