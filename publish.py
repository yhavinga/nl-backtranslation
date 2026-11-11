"""The two text checks of FLAN-nl's publish.py that pairs.py and mainset.py use: decode_loop (a translation that
repeats a sequence its source does not repeat) and english_left (a stretch of the source language left in the
translation), with their word lists. Copied unchanged from publish.py of FLAN-nl (github.com/yhavinga/flan-nl); the
rest of that file builds the FLAN-nl dataset and is not needed here."""

import re

# The checks of this publication: two failures that flan_nl.py's checks let through, found in samples of the first
# build (9 Nov), decode loops in long passages and English left untranslated (langid takes a text that is mostly Dutch
# for Dutch). They only decide what goes into this dataset; F-nl's training data stays as its protocol fixed it. Each
# threshold was chosen by reading random samples of the rows it flags and of the rows just below it (HANDOVER.md).
TOKEN = re.compile(r"\w+|[^\w\s]")
LETTERS = re.compile(r"[^\W\d_]+")
# A sequence of 1-4 tokens 5 or more times back to back; at 4, an ellipsis and a blank ("... .", "_ .") are among them
LOOP_SHORT = (range(1, 5), 5)
LOOP_LONG = (range(5, 17), 3)  # of 5-16 tokens 3 or more times
# A translation can lengthen the sequence ("0 0 0" -> "0, 0, 0"), so the English's may be twice as long
LOOP_ENGLISH_PERIODS = range(1, 33)
LOOP_ENGLISH_COPIES = 3
# English function words, other lowercase English words of 3 letters or more ("'s" and "'re" are not): titles have
# few of the second; at 4 of them, quoted titles and lyrics were a quarter of a sample
ENGLISH_LEFT = (3, 5)
# English words that are not Dutch, or only rare Dutch
ENGLISH_WORDS = frozenset("""
    the and to that for with it are this on by from at not have has you your which they their them would can what who
    whom whose there were these those should could about into than then also its his she him her he when where while
    will after before because being does did said other some such only any each very just how why our my but if or so
    up out all get got more most many much like make made yes""".split())
# Dutch function words: English left in a Dutch sentence ends at one
DUTCH_WORDS = frozenset("""
    de het een en van dat die niet zijn op te met voor er maar om ook bij door naar dan nog wel uit ze hij zij wat wordt
    worden werd werden kan kunnen zou zouden heeft hebben hun haar hem ik je jij jou jouw u uw wij deze dit daar hier
    waar wie welke welk omdat toen nadat voordat terwijl tijdens tegen onder tussen na zonder geen alle moet moeten mag
    kon konden waren ben bent zal zullen wil willen als aan sinds zich mijn ons onze iets niets veel meer ja nee hoe
    waarom wanneer dus want toch echter alleen nu""".split())
# Words of both languages, evidence of neither
SHARED_WORDS = frozenset("""
    a an i in is was had of over me we as man hand arm warm been kind room test team film films hotel park bus radio
    computer sport water winter wind plan hard best even ten top net nest mild slim lip land open start stop week
    museum school""".split())
# Copied as they are in a translation, not text to translate: URLs, e-mail addresses, @names, #tags
NOT_PROSE = re.compile(r"(?:https?://|www\.)\S+|\S+@\S+\.\w+|[@#]\w+")
_INNER = re.compile(r"(.{2,}?)\1{2,}")


def _is_word(token):
    return token[0].isalnum() or token[0] == "_"


def repeats(tokens, periods):
    """The most copies of a sequence of p tokens (p in periods) with a word in it, back to back; 1 where none repeats.
    Sequences of punctuation only do not count: in a sample they were tables the translation drew (`| ------- |`) and
    emoji the English has as HTML entities, hardly ever a decode loop."""
    import numpy as np

    best = 1
    if len(tokens) < 2:
        return best
    vocab = {}
    ids = np.fromiter((vocab.setdefault(t, len(vocab)) for t in tokens), np.int64, len(tokens))
    words = np.concatenate(([0], np.cumsum([_is_word(t) for t in tokens])))
    for p in periods:
        if len(ids) <= p:
            break
        # A run of k tokens equal to the token p before them is k // p + 1 copies of a sequence of p tokens
        same = np.concatenate(([False], ids[p:] == ids[:-p], [False]))
        edges = np.flatnonzero(same[1:] != same[:-1])
        if not len(edges):
            continue
        starts, lengths = edges[::2], edges[1::2] - edges[::2]
        has_word = words[starts + p] - words[starts] > 0
        if has_word.any():
            best = max(best, int((lengths[has_word] // p + 1).max()))
    return best


def _inner_repeats(tokens):
    """The most copies of a piece back to back inside one token: emoji written out ("loudlycryingface" four times in
    one word), which a translation turns into separate words"""
    best = 1
    for t in tokens:
        if 6 <= len(t) <= 300:
            for m in _INNER.finditer(t):
                best = max(best, len(m.group()) // len(m.group(1)))
    return best


def decode_loop(nl, en):
    """The translation nl repeats a sequence back to back where its English en has no such repeat.

    In nl (lowercase tokens: words and punctuation), a sequence with a word of LOOP_SHORT's lengths at least
    LOOP_SHORT's copies, or of LOOP_LONG's at least LOOP_LONG's, also counted over the words alone, as decode loops
    vary their punctuation ("for, for: for: for,"). The English excuses it with a repeat of at least
    LOOP_ENGLISH_COPIES and half as many copies, counted the same way and also inside one token: "No emotion,No
    emotion,..." -> "Geen emotie, geen emotie, ...", "goals: 0 0 0 0 0" -> "doelpunten: 0, 0, 0, 0, 0"."""
    tn = [t.lower() for t in TOKEN.findall(nl)]
    words = [t for t in tn if _is_word(t)]
    found = []
    for periods, least in (LOOP_SHORT, LOOP_LONG):
        n = max(repeats(tn, periods), repeats(words, periods))
        if n >= least:
            found.append(n)
    if not found:
        return False
    te = [t.lower() for t in TOKEN.findall(en)]
    english = max(repeats(te, LOOP_ENGLISH_PERIODS), repeats([t for t in te if _is_word(t)], LOOP_ENGLISH_PERIODS),
                  _inner_repeats(te))
    return any(english < max(LOOP_ENGLISH_COPIES, n / 2) for n in found)


def english_left(nl, en):
    """The translation nl has a stretch of its English en in it: a run of words of en, none a Dutch function word,
    with at least ENGLISH_LEFT[0] English function words and ENGLISH_LEFT[1] other lowercase words of 3 letters or
    more that are not words of both languages.

    Names and titles are copied too, but are capitalized and have few lowercase words ("The Lord of the Rings"); a
    Dutch word that en does not have ends the run. Words in capitals count for nothing; URLs and the like are left
    out."""
    en_words = {w.lower() for w in LETTERS.findall(NOT_PROSE.sub(" ", en))}
    least_function, least_other = ENGLISH_LEFT
    function = other = 0
    for w in LETTERS.findall(NOT_PROSE.sub(" ", nl)):
        if not (w.islower() or w.istitle()):
            continue
        lw = w.lower()
        if lw in DUTCH_WORDS or lw not in en_words:
            function = other = 0
        elif lw in ENGLISH_WORDS:
            function += 1
        elif w.islower() and len(w) >= 3 and lw not in SHARED_WORDS:
            other += 1
        if function >= least_function and other >= least_other:
            return True
    return False
