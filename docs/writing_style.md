Style for docs, docstrings and comments
=======================================

The documentation, docstrings and code comments in sotrplib use the rules of
ASD-STE100 Simplified Technical English (STE), <https://www.asd-ste100.org>.
STE makes text easier to read, also for readers who do not speak English as a
first language. This page gives the rules that apply to this repository.

The approved words are in the STE dictionary (Part 2 of the specification).
You can get the specification free of charge from the ASD-STE100 website. If
you are not sure about a word, look in the dictionary or use a simpler word.


Words
-----

1. Use approved words only. Use each word only with the sense and the part
   of speech that the dictionary gives. For example, "use" is approved, but
   "utilize" and "leverage" are not.
2. You can use technical names and technical verbs. In this repository, these
   include:
   - code identifiers, in backticks: `sotrp-coadd`, `stream_coadd()`,
     `track_processing`, `map_id`;
   - field and status values, in backticks: `rho`, `kappa`, `completed`,
     `permafail`;
   - domain terms: coadd, depth-1 map, matched filter, pointing model, flux,
     inverse variance, SNR, SLURM, FITS, mapcat, SOCat.
3. Use one word for one object. Do not use synonyms. For example, do not call
   one object a "coadd", a "stack" and a "combined map" on the same page.
4. Do not use more than three nouns in a noun cluster. Change "coadd output
   directory root path" to "the root directory for coadd outputs".
5. Do not use contractions. Write "does not", not "doesn't".
6. Do not use Latin abbreviations. Write "for example" (not "e.g."), "that is"
   (not "i.e."), and "through" (not "via"). Do not use "etc.".
7. Do not use phrasal verbs. Write "continue" (not "carry on"), "start" (not
   "kick off"), "find" (not "figure out").


Verbs
-----

1. Use only the simple present, simple past and simple future tenses.
2. Use the active voice. In descriptive text, you can use the passive voice
   only when the agent is not known or not important.
3. Do not use the "-ing" form of a verb, except in a technical name. Change
   "Before merging the map, ..." to "Before the function merges the map, ...".
4. Use "must" for a requirement. Do not use "should", "shall", "need to" or
   "have to".


Sentences and paragraphs
------------------------

1. Write a maximum of 20 words in a sentence of an instruction (a procedure).
2. Write a maximum of 25 words in a sentence of a description.
3. Write only one instruction in a sentence. Use the imperative: "Set
   `--time` to 08:00:00."
4. Put the most important information first.
5. Write only one topic in a paragraph. Write a maximum of six sentences in a
   paragraph.
6. Use a vertical list (bulleted or numbered) for three or more items, and a
   numbered list for steps that the reader must do in sequence.
7. Keep the articles ("a", "an", "the") and "that". Do not remove words to
   make a sentence shorter.


Warnings and cautions
---------------------

Start a warning or a caution with a clear command. Then give the reason.

- Bad: "Note the 4 h default is too short for busy weeks."
- Good: "Set `--time` to 08:00:00 for busy weeks. The default time limit of
  4 hours is too short for 50 maps."


Docstrings
----------

- Start with one sentence in the present tense that tells what the function
  does: "Return the coadd of the maps that the reader selects."
- For a parameter, give its meaning, its unit and its default.
- For an exception, write the condition: "Raises `ValueError` if every input
  map fails."

Example, from `stream_coadd()` in `sotrplib/maps/map_coadding.py`:

```python
# Before
    """
    Memory-bounded coadding: build, preprocess (e.g. matched filter), and
    merge `maps` into a single coadd one at a time, discarding each raw map
    before moving on to the next, rather than holding every input map in
    memory at once.
    """

# After
    """
    Build a coadd from `maps`, one map at a time.

    For each map, the function builds the map, applies the preprocessors
    (for example, the matched filter) and merges the map into the coadd.
    Then it removes the map from memory. Thus, memory use does not increase
    with the number of maps.
    """
```


Comments
--------

- Tell why the code does an operation. Do not repeat what the code tells.
- Use full sentences, with the same rules as the docs.
- Do not write comments that describe the history of a change ("now uses X
  instead of Y"). Put that information in the commit message.

Example, from `sotrplib/coadd_cli.py`:

```python
# Before
# A coadd's id doesn't exist until register_coadd() has already
# finished, so unlike maps (which get set_processing_start()
# before we know if they'll succeed), a coadd's status row is
# only created once the outcome is known. set_processing_end()
# requires an existing row, so create it first.

# After
# The coadd gets its coadd_id only after register_coadd() is complete.
# Thus, the coadd gets its status row after the result is known.
# set_processing_end() needs a row, so create the row first.
```
