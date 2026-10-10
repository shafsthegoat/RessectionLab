# Whole-episode instrument camera fit

Live contact-family replay exposed a clipped long probe: camera fitting happened before the replay installed its instrument, and subsequent tool switches used only the current frame. The viewer now derives a full-episode capsule envelope after replay validation and fits it once when a new episode appears. Timeline changes preserve the user's orbit and zoom; manual Fit covers the entire episode.

Five canonical geometry controls and all 77 viewer controls pass. They project the complete capsule bounds at three aspect ratios, exercise actual saved generated replay, test tool switches and reject mixed identities. The desktop build passes; independent source review agrees. No planner, physics, checkpoint or input-admission change. Live verification of this repair follows separately; the original clipping evidence remains preserved.
