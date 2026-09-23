# ComicReel — What's Been Done (Simple Version)

This is a plain-English summary of everything done so far. For the detailed, timestamped version, see `timeline.md`.

## The idea
ComicReel is a tool that will turn scanned comic books into little animated movies — with voices, music, and sound effects — using free AI tools.

## What we've done so far

1. **Set up the project folders.** Made a proper folder structure for the code (`src/`), tests (`tests/`), settings (`configs/`), and scripts (`scripts/`).

2. **Wrote the skeleton code.** Wrote the basic building blocks of the program — not the AI parts yet, just the "plumbing":
   - A settings file that says what the program should do (`pipeline.yaml`)
   - Code that reads those settings
   - Code that remembers work already done, so it doesn't redo it
   - A command-line tool (`comicreel`) you can type in the terminal to run things
   - Placeholder spots for each of the 10 stages of the pipeline (panel detection, bubble detection, reading dialogue, generating a script, text-to-speech, animation, music, and putting it all together into a video)

3. **Wrote down the plan.** Made a `progress.md` file that lists all 10 phases of the project and tracks which parts are done and which aren't, so we always know where we stand.

4. **Got the tools working.** The first time we set this up, we couldn't install anything because of no internet. Today the internet was working, so we:
   - Rebuilt the project's Python environment (it was using the wrong Python version before)
   - Downloaded and installed the basic packages needed (things like `typer`, `pydantic`, `pytest`, `ruff`)
   - Ran the program and its tests to make sure everything actually works — and it does

5. **Cleaned up the code style.** Ran a tool called `ruff` that checks for messy or outdated code and fixed the small issues it found.

6. **Tidied up the folder.** Deleted leftover junk files that get auto-created when you run tests or checks (they're not needed and get recreated automatically anyway).

7. **Started keeping a history log.** Made a `timeline.md` file that records every single file that gets created, changed, or deleted, so nothing is forgotten.

## Phase 1: teaching the program to find panels

8. **Got a real AI model working.** Downloaded a free AI model called "Magi" (about 2GB) that's specifically trained to look at a comic page and find the individual panels, speech bubbles, and characters. Had to fix a few compatibility hiccups along the way (missing packages, a version mismatch) before it would actually run.

9. **Also built a simpler backup method.** Wrote a second, much simpler way to find panels that doesn't need any AI model at all — it just looks for the white/blank gaps between panels (like following the empty margins on a page) and uses those to figure out where each panel is. This one is instant and works with zero downloads, as a fallback for when the AI model isn't available or needed.
   - First attempt at this simpler method didn't work well on a real scanned comic page (it got confused by faded panel borders). Rewrote it with a better technique and it then found every panel correctly.

10. **Taught it what order to read panels in.** Comics aren't read top-to-bottom like a text document — panels need to be read left-to-right (or right-to-left for manga) row by row. Wrote a dedicated piece of code for this so both the AI model and the simple backup method read panels in the correct order.

11. **Got real comic pages to test on.** Downloaded 3 pages from a real, legally free (public domain) 1946 comic book to use as test material, since made-up test images wouldn't prove the panel-finder actually works on real scans.

12. **Checked the results by eye.** Drew red boxes with numbers around every panel the program found, on all 3 real test pages, for both methods (the AI model and the simple backup) — 23 panels total across the 3 pages, every single one found correctly in the right reading order.

13. **Made the command-line tool actually do something.** Previously, typing a command like "find the panels on this page" just printed a fake placeholder message. Now it actually runs the panel-finder and prints the real result.

14. **Wrote automatic tests.** Added 11 new automatic checks (on top of the 2 from before) that run in a couple seconds and catch it if this code ever breaks later — 13 tests total, all passing.

15. **Made a folder to look at the results.** Created a `phase 1 output` folder with the AI model's actual results for all 3 test pages — both a picture with red boxes drawn around each panel, and the raw computer-readable data behind it.

16. **You spotted a mistake, and we fixed it.** Looking at the results, you noticed the box around the very last panel looked a bit off. Turned out you were right — that panel had a caption at the top ("GOOD -- NOW LET'S HAUL THESE SACKS...") and the AI's box was cutting off part of it, starting about 30 pixels too low. Traced it to the AI getting confused by a small bright gap inside the caption box, mistaking it for the actual panel edge. Fixed it by adding a check that only trusts a "gap" as a real panel edge if it's wide enough (at least 8 pixels) — narrow gaps like the one inside the caption get ignored now. Regenerated the results and confirmed every box is now correctly sized on all 3 pages, including the one that was wrong. Added tests for this fix too (17 tests total now).

## Phase 2: finding speech bubbles and figuring out who's talking

17. **Reused the same AI model for a new job.** Turns out the "Magi" AI model from Phase 1 doesn't just find panels — in the very same step, it can also find speech bubbles, caption boxes, "tails" (the little pointer that connects a speech bubble to whoever's talking), and where characters are standing on the page. So instead of downloading a second AI model, the program now just asks Magi for more of what it already gives back.

18. **Made the program smart enough to guess who's speaking.** Wrote logic that looks at which way a speech bubble's tail is pointing and figures out which character it's pointing at — like following an arrow. If a bubble has no tail (usually a narration caption, not speech), it just picks whichever character is standing closest.

19. **Made sure the loaded AI model doesn't get loaded twice.** Since both the panel-finder and the bubble-finder now use the same 2GB AI model, rewrote the code so it's only loaded into memory once and shared, instead of being loaded separately by each one (which would waste time and memory).

20. **Checked the results by eye and caught a real bug.** Looked at a picture with the bubbles and "who's talking" lines drawn on it, and found one caption box was being matched to a character all the way on the other side of the page — because the guessing logic was searching the *whole page* for the closest character instead of just the panel the bubble is actually in. Fixed it so it only looks within the same panel now, and confirmed the lines now stay sensible.

21. **Found (and understood, rather than papering over) a real limitation.** On one of the 3 test pages, the AI model reported every single bubble as a caption instead of a speech bubble, even though some visibly have tails drawn on them. Investigated and found the AI just wasn't confident enough about which tail belongs to which bubble on that specific page — tested lowering the confidence bar and found that doing so would just make the model guess wildly on *every* page instead, causing new mistakes elsewhere. Decided it's better to leave the setting as the AI's own recommended default and note this as a known limitation, rather than force-fixing one page at the cost of accuracy everywhere else.

22. **Wrote more automatic tests.** Added 9 more tests (8 for the "who's talking" guessing logic, 1 that checks the bubble-finder's output against the expected format) — 26 tests total now, all passing.

23. **Made a results folder for this phase too.** Same idea as the `phase 1 output` folder — created `phase 2 output` with pictures (bubbles boxed in blue, tails in green, "who's talking" lines in yellow) and raw data for all 3 test pages: 10, 14, and 11 speech/caption bubbles found respectively.

## Phase 3: actually reading the dialogue and describing what's happening

24. **Checked how much disk space we have left before downloading anything big.** Only about 30GB free. The AI model originally planned for "describe what's happening in this panel" (a 7-billion-parameter model) would take up half of that on its own — too risky.

25. **Gave you a way to test a few AI models yourself before committing to one.** Instead of picking based on specs alone, wrote a test prompt and instructions so you could try 4 candidate models (of different sizes) on our 3 sample pages yourself, using their free online demos — no downloads needed just to try them. Waiting on your pick.

26. **Kept working on everything that doesn't need that download.** You asked for this specifically, so:
    - **Got dialogue-reading (OCR) working with zero new downloads.** Turns out the AI model from Phase 1 (Magi) already secretly includes a small English text-reading model as part of itself — so instead of downloading a whole new OCR tool, the program just asks Magi to also read the text out loud (well, into words). Also wired up "Tesseract," a free text-reader that was already installed on this computer.
    - **Tested both text-readers on real dialogue and measured which is better.** Neither one is perfect — one sometimes badly misreads fancy comic lettering (turning "HEH-HEH-HEH" into "HEH-NEW-NEW", or a caption into a fake web address), the other sometimes returns almost nothing at all, but claims to be very confident it's right anyway. So rather than trusting one over the other blindly, the program now runs both and keeps whichever one actually recovered more real words — tested on real comic text, and it correctly avoids every failure case we found.
    - **Prepared everything for whichever "describe the scene" AI model you pick.** Wrote the exact instructions (prompt) that'll be sent to whichever model you choose, plus the code that reads its answer back out reliably (even if the AI wraps its answer in extra chit-chat, which AI models often do).
    - **Double-checked no AI model got secretly downloaded.** Confirmed the project's saved-models folder is still exactly the same size as before — nothing new was downloaded, as asked.

27. **Wrote more automatic tests.** Added 20 more tests for all of the above — 46 tests total now, all passing.

28. **You got more free disk space, so we picked a real AI model.** Once space wasn't tight anymore, you asked for an honest, sourced recommendation instead of a guess. Looked up real test scores comparing a bigger version of the AI model against the one we were considering — turned out the smaller one actually wins specifically at reading text inside pictures (which matters a lot for comic panels with lettering baked into the art). Recommended it with the receipts to back it up, and you picked it: **Qwen2.5-VL, the 7-billion-parameter version.**

29. **Downloading it turned into a whole saga.** In order:
    - The first download attempt "hung" for hours with no visible progress. Turned out data actually was flowing the whole time — the downloading tool just doesn't show progress until it finishes one giant multi-gigabyte chunk, so it looked frozen when it wasn't.
    - Separately, your laptop went to sleep overnight while the download was running, which paused it completely for hours without any warning — a much bigger, silent time-waster than the tool's fake "stuck" appearance.
    - Fixed both: switched to a different, more transparent way of downloading, and from then on told the computer to stay awake while anything long-running was in progress.
    - The connection dropped once more partway through (at 15 out of 16 GB) — but resuming just picked up where it left off instead of starting over.
    - Eventually, the full ~15GB file finished downloading.

30. **You ran it once, and it was still way too slow — so we found the real reason and fixed it properly.** You pointed out, fairly, that your other AI models (through a different tool called Ollama) load in seconds, so something had to be wrong. Dug into it and found genuine reasons: the version we downloaded was the full, "uncompressed" version of the model (about 3x bigger than what Ollama normally uses), and Ollama itself uses much more optimized, Mac-tuned software to run models than what we were using. So we **switched the whole approach**: deleted the big 15GB file, downloaded the same AI model again but in Ollama's much smaller, optimized format (6GB instead of 15GB), and rewrote the code to talk to Ollama instead. As a bonus, this new approach also guarantees the AI always answers in the exact format we need, instead of us having to double check its homework.

31. **Cleaned up after the switch.** Once we confirmed the Ollama approach worked, deleted the old 15GB download to free up space, and removed the leftover backup code for the old approach entirely, instead of leaving unused stuff lying around that could cause confusion later.

32. **It finally worked — properly, end to end.** Ran it for real on a whole comic page (7 panels): read out what's happening in each panel, described the setting and mood, spotted sound effects like "HEH-HEH-HEH", and combined it all with the dialogue text — all in under 10 minutes, compared to the earlier attempt that couldn't even finish one panel in hours. It's not perfect (a 7-billion-parameter AI model does make mistakes — at one point it mixed up which mouse character was which), but overall the results made sense and matched what's actually happening in the comic.

33. **Did this for all 3 test pages and saved the results**, same as the previous two phases — a `phase 3 output` folder with the full read-out for each page.

## Where things stand now
Phase 0 (project setup), Phase 1 (finding panels + reading order), Phase 2 (finding speech bubbles + guessing who's talking), and now Phase 3 (reading dialogue + describing each panel) are all fully done and verified on real comic pages. Next up: figuring out who's who across the whole comic (Phase 4), then writing the script, generating voices, and making a video.
