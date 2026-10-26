"""
AI Health Monitor - Sentence Bank (Module 18)
=============================================

Neutral English sentences for the 10-line behavioural typing assessment.
Content rules: no sensitive, political, medical, religious or personal data.
The bank is the SINGLE source of truth - the assessment page fetches a random
selection through /api/sentences, so no two sessions see the same 10 lines.

PRIVACY: sentences are test stimuli, not user content. What the user types is
never transmitted - only timing metadata and aggregate counters.
"""

import random

SENTENCE_BANK = [
    "The morning sunlight slowly filled the room with a warm golden glow.",
    "Technology can make everyday tasks easier when it is designed thoughtfully.",
    "The quick brown fox jumps over the lazy dog near the quiet forest.",
    "The quiet river reflected the trees as the evening sky slowly became darker.",
    "Learning new skills becomes easier when practice is consistent and enjoyable.",
    "A reliable system should remain simple enough for people to understand and use.",
    "Small changes in daily behaviour can sometimes reveal meaningful patterns over time.",
    "Good software should provide useful information without interrupting the user's normal workflow.",
    "She watered the balcony plants while the city hummed quietly below her window.",
    "The library was almost empty except for the soft sound of turning pages.",
    "Fresh bread from the corner bakery always draws a short patient queue.",
    "A calm walk in the evening helps the mind settle after a busy day.",
    "The train arrived two minutes early and the platform slowly came to life.",
    "Painters often mix their own colours to capture a very specific shade of light.",
    "The old clock on the wall kept perfect time despite its worn wooden case.",
    "Children were flying a kite near the hill while their dog chased shadows.",
    "Good notes turn a messy brainstorm into a plan the whole team can follow.",
    "The garden smelled of rain because the clouds had just passed over the village.",
    "A balanced breakfast is a simple way to start a long morning of work.",
    "The museum guide explained how the ancient tool was carved from a single stone.",
    "Musicians tune their instruments carefully before the first rehearsal begins.",
    "The mountain trail offered a wide view of the valley after the final bend.",
    "Typing comfortably depends more on posture and rhythm than on raw speed.",
    "The coffee shop played soft jazz while laptops glowed at every table.",
    "Farmers checked the sky each evening because weather shaped the whole week.",
    "A tidy desk makes it easier to find the right tool at the right moment.",
    "The lighthouse swept its steady beam across the water every few seconds.",
    "Good questions open doors that long explanations often leave closed.",
    "The bridge lit up at dusk and reflected in the river like a string of lanterns.",
    "Reading a few pages before sleep became her favourite quiet ritual.",
    "The workshop smelled of cedar shavings as the carpenter smoothed the drawer.",
    "A short break every hour keeps both focus and energy steady through the day.",
    "The puppy learned to sit after three patient afternoons of gentle practice.",
    "Autumn turned the park into a carpet of amber, rust and fading green.",
    "The recipe called for patience more than skill, and the result was worth it.",
    "City birds started singing an hour before the first bus rumbled past.",
    "The telescope revealed a cluster of faint stars beyond the familiar constellations.",
    "A well written summary saves every reader several minutes of careful searching.",
    "The bakery's chalkboard announced cinnamon rolls as the special of the day.",
    "Slow breathing before a presentation steadies both voice and thoughts.",
    "The ferry cut smoothly across the bay while gulls circled behind it.",
    "Careful listeners often notice the details that impatient ones skip entirely.",
    "The chess match lasted an hour and ended in a friendly, hard earned draw.",
    "Morning fog lifted slowly as the first sunlight reached the harbour.",
    "The notebook was filled with small sketches of doors, windows and staircases.",
    "A single candle made the whole room feel warmer during the power cut.",
    "The hiking group shared oranges at the summit before starting back down.",
    "Good habits grow quietly, one small repeated action at a time.",
    "The train station clock glowed softly above the nearly empty platform.",
    "Winter light through frosted glass painted pale shapes across the floor.",
    "The recipe box held three generations of handwritten family favourites.",
    "A gentle rain began as the class finished its sketches of the courtyard.",
    "The old theatre restored its velvet curtains and gilded balcony rails.",
    "Quiet mornings are perfect for planning the day without any hurry.",
    "The potter shaped the clay with steady hands and a focused expression.",
    "Lanterns floated above the festival grounds, drifting slowly toward the hills.",
    "The atlas showed tiny islands that most maps choose to leave unnamed.",
    "A cup of tea and a good book make a simple evening feel complete.",
    "The cyclists coasted down the long hill with the wind rushing past them.",
    "Careful pacing helped the runner finish strong instead of burning out early.",
    "The greenhouse stayed warm and green despite the frost outside its glass.",
    "The baker scored each loaf once before sliding it into the stone oven.",
    "Snow softened every corner of the village and hushed the usual evening sounds.",
    "The orchestra tuned as the audience settled into their seats with quiet anticipation.",
    "A morning checklist keeps important tasks from slipping into the afternoon.",
    "The lantern in the hallway flickered once and then burned steadily again.",
    "Fresh sheets of paper waited beside the ink bottle and the brass pen.",
    "The boys mapped their secret trail with a pencil and a borrowed compass.",
    "Cool evening air carried the scent of pine down from the ridge above town.",
    "The potter's wheel hummed as a simple lump became a graceful bowl.",
    "A short stretch at the desk eases the shoulders after long concentration.",
    "The harbor seals dozed on the warm rocks while fishing boats motored out.",
    "Steady practice turned the difficult chord into something her fingers found easily.",
    "The city skyline glowed softly as the last daylight slipped behind the hills.",
]


def random_sentences(n=10, rng=None):
    """Return `n` distinct random sentences from the bank (shuffled order)."""
    rng = rng or random
    n = max(1, min(int(n), len(SENTENCE_BANK)))
    return rng.sample(SENTENCE_BANK, n)
