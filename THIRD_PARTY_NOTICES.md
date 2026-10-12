# Third-party data and licenses

The MIT license in this repository covers the original PitchIQ code, documentation,
and demo assets. It doesn't cover football providers' data, club badges or other
trademarks, or the Python packages used by the project.

The public app can fetch **OpenFootball CC0** match results when someone chooses
real data on the match pages. It doesn't commit those results or any pretrained
real-data models to GitHub. The real match models are trained in memory.
The player and shot tools in the public app use fictional data. They do not
silently switch to research datasets.

I tried several data sources while building PitchIQ. Some are available to read
but not necessarily free to republish or train on, so I've kept those datasets
and their model files out of this public repository. The links and notes below
record the distinction. This isn't legal advice; check the actual terms before
reusing or distributing data.

## Optional research sources

Terms checked October 11, 2026. Review the actual terms for your use; upstream
terms and repository ownership can change.

| Source | Evidence | Public-export policy |
| --- | --- | --- |
| OpenFootball match JSON | [football.json CC0 license](https://github.com/openfootball/football.json/blob/master/LICENSE.md) | Documented CC0 permission; opt-in match mode fetches results at runtime, not bundled. |
| OpenFootball Champions League text | [champions-league CC0 license](https://github.com/openfootball/champions-league/blob/master/LICENSE.md) | Same CC0 opt-in fetch policy. CC0 does not grant trademark rights. |
| StatsBomb Open Data | [Public Data User Agreement](https://github.com/hudl/open-data/blob/master/LICENSE.pdf), [README](https://github.com/hudl/open-data/blob/master/README.md) | Proprietary terms restrict data redistribution and commercial exploitation; published analysis requires credit and the official logo. Exclude raw events, extracted shots, real-data-trained models, and related images pending rights confirmation. This is not a Creative Commons dataset. |
| Transfermarkt community downloads | [dcaribou CC0 declaration](https://github.com/dcaribou/transfermarkt-datasets/blob/master/LICENSE), [Transfermarkt terms](https://www.transfermarkt.co.uk/intern/anb), [legal notice](https://www.transfermarkt.co.uk/intern/impressum) | The intermediary's CC0 declaration does not clear someone else's rights. Current provider terms restrict automated extraction and AI/ML use; the legal notice restricts reproduction. Exclude original/cleaned tables and related artifacts. |
| Understat community aggregation | [source repository](https://github.com/vibedatascience/understat_players_aggregated), [Understat](https://understat.com) | No license was established for the aggregation. Exclude original/cleaned tables and related artifacts pending permission. |
| Legacy `data/players.csv` | Provenance not established | Excluded. |

These are release decisions based on the available source documents, not a legal
opinion that every possible use is authorized. An MIT license cannot clear rights
you do not own. Obtain appropriate permission before adding external data or
publishing a live research mode. If any source code was copied from elsewhere,
its original license and notices must also be honored.
