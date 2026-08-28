# Acceptable use

This project enlarges and restores images on your own machine. Every mode it ships
reconstructs: it recovers detail the pixels still imply. None of them is built to invent
detail — though an adversarially trained model can still emit texture the source did not
contain, which [`README.md`](README.md#reconstruction-only) and
[current limitations](docs/reference.md#current-limitations) both record. That boundary is
deliberate, and this file states what follows from it.

This is not a licence condition. The [Apache 2.0 licence](LICENSE) grants what it grants
and this document does not add restrictions to it. It is the maintainer's statement of
intended use, the reasoning behind the defaults, and the standard contributions are held
to.

## The one rule

**Do not use this software to depict a real person in a way they have not agreed to.**

Concretely, do not use it to produce:

- Sexual or intimate imagery of a real person without their explicit consent, or of
  anyone who is or appears to be a minor, under any circumstances.
- Imagery that presents a real person as having said, done, or been somewhere they were
  not, where it could be mistaken for a record of them.
- "Recovered" or "enhanced" images offered as identification, evidence, or proof about a
  person — in a legal proceeding, an investigation, a news report, or an accusation.
- Alterations to a person's body, face, age, or appearance that are passed off as a
  photograph of them.

Consent means the depicted person agreed to this use. Access to a photograph of someone
is not consent. Neither is the photograph having been public.

## An upscale is still not evidence

Reconstruction is a weaker claim than generation, not a harmless one. A super-resolution
model trained on millions of photographs infers what a soft edge or a coarse texture most
likely was; on a face forty pixels wide it has almost nothing to work from, and what it
produces is a statistical guess that happens to look photographic.

So the rule above still governs the output. Enlarging a licence plate, a face in a crowd,
or a distant sign produces something that looks sharper and is not a better observation
than the original. Nothing this application returns is an identification of a person or
proof of what a picture showed.

The interface says so where it matters: the exact output dimensions, the engine that ran,
the enlargement factors that were chained, and a warning wherever an inferred result could
be mistaken for a recovered one. Those labels are load-bearing.

## Generation is out of scope

Diffusion restoration, face priors, prompt-driven editing, and any other stage that invents
detail rather than recovering it are not part of this project. That is a scope decision
rather than a disabled feature: there is no flag that turns them on, and the test suite
asserts that no engine and no mode here claims to be generative.

A change that adds one will be declined, for the reason set out above: a generated face is
an output of a model, not an observation of a person, and the distance between those two
things is exactly what this design exists to protect.

## Reporting misuse

This is local-first software with no telemetry and no server: there is no account to
suspend and no content for a maintainer to take down. If you find this project being used
against someone, the useful routes are the platform hosting the material and, for
non-consensual intimate imagery, [StopNCII.org](https://stopncii.org) (adults) or the
[NCMEC CyberTipline](https://report.cybertip.org) (minors).

Security vulnerabilities are a separate matter; see [`SECURITY.md`](SECURITY.md).

## For contributors

Changes are declined if they present a resample as recovered detail, remove or weaken a
label that keeps an inferred result distinguishable from a recorded one, or add a
capability that synthesises detail the source never contained. See
[`AGENTS.md`](AGENTS.md) for the engineering rules these follow from, and
[`CONTRIBUTING.md`](CONTRIBUTING.md) for how to get a change reviewed.
