# FOTO-NET site design research

This brief records why the site looks and reads the way it does. It is not
public product documentation.

## The recognizable “AI slop” pattern

Current critiques consistently identify convergence on the statistical average
of recent SaaS marketing sites: a centered hero, vague superlative headline,
purple/blue gradient, glass or soft-shadow cards, excessive rounding, three
generic benefits, decorative dashboard mockups, interchangeable sans-serif
typography, fake customer logos/testimonials, and repeated low-density sections.
The surface is polished but carries little evidence of the actual product.

The deeper problem is not any individual gradient or rounded corner. It is a
lack of decisions. Every section uses the most probable layout; copy promises
outcomes without naming constraints; visual hierarchy repeats rather than
responding to content; and “proof” is detached from a reproducible source.

Sources consulted:

- Shuffle, “Why Do Most AI-Generated Websites Look the Same?”
  https://shuffle.dev/blog/2026/01/why-do-most-ai-generated-websites-look-the-same/
- Joshua Snoddy, “Why Do AI-Generated Websites All Look the Same?”
  https://www.joshuasnoddy.com/blog/why-ai-websites-look-the-same/
- Curio, “Why AI-Generated Design All Looks the Same”
  https://designbycurio.com/learn/why-ai-design-looks-generic
- Luke Wroblewski, “How AI Ate My Website”
  https://static.lukew.com/HowAIAteMyWebsiteFeb2024.pdf
- “The Effects of Generative AI on Design Fixation and Divergent Thinking”
  https://arxiv.org/abs/2403.11164

These essays are observations rather than design standards. The standards and
documentation structure below supply the operational rules.

## Rules applied here

1. **Start with the artifact, not a template.** The first viewport identifies
   Architecture 4 / Foundation 2 / Head 4, current alpha status, raw output,
   deploy parameters, and the exact boundary of the speed result.
2. **Never fabricate proof.** No testimonials, company-logo strip, download
   count, AP value, pretrained-weight claim, or vague “state of the art” label.
3. **Use the correct information shape.** Tables compare models; definition
   lists describe contracts; ordered sequences explain the graph; prose carries
   caveats. Cards are not the universal container.
4. **Make one visual argument.** The site resembles a field manual and lab
   record: ruled borders, indexed sections, editorial serif headings, compact
   monospace metadata, warm paper, and technical green.
5. **Keep geometry disciplined.** Square corners, visible alignment lines,
   asymmetric 60/40 layouts, and limited shadows. No glow blobs, glass panels,
   floating orbs, giant empty hero, or decorative fake application UI.
6. **Let density vary with purpose.** The home page is an editorial overview;
   documentation pages use a persistent manual index; benchmarks use a wide
   numeric table and explicit methodology.
7. **Color has roles.** Every theme replaces background, paper, ink, muted text,
   borders, accent, signal, code surface, selection, and shadow tokens. Themes
   are not single-accent recolors.
8. **Motion is functional and scarce.** Only state transitions and hover
   feedback; reduced-motion preferences remove them.
9. **Copy makes falsifiable statements.** “218 FPS” is always tied to device,
   precision, input, graph state, and exclusions. Unknown AP remains unknown.
10. **Documentation is a product surface.** Install, inference, training,
    resume, models, export, Results, transforms, security boundaries, and
    benchmarks are available inside the site—not delegated to a GitHub link.

## Documentation architecture

The information model follows the Diátaxis distinction between tutorials,
how-to guides, reference, and explanation: https://diataxis.fr/. The website
does not create empty categories for symmetry; each route names its actual mode.

Accessibility decisions follow W3C WAI guidance: sufficient contrast, no
color-only status, recognizable interactive elements, consistent navigation,
semantic headings, responsive layouts, keyboard focus, and reduced-motion
support: https://www.w3.org/WAI/tips/designing/.

## Palette directions

- **Field** — default warm beige paper, deep botanical green, terracotta signal.
- **Night lab** — dark chlorophyll surfaces, bone text, celadon accent.
- **Blueprint** — warm drafting paper, technical blue-green, rust signal.

All three retain the same hierarchy and content. The switch is a user
preference stored locally on the device.
