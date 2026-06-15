# Pipeline Flowchart

The workflow for the time-resolved pipeline.
Inputs are read from a range of databases and written to the same(???) databases (green).
Gray boxes are objects that are stored in memory.

```mermaid
---
config:
  layout: elk
---
flowchart TD

initial@{ shape: procs, label: "Initial maps"}
coadded(Coadded map)
processed(Processed map)
fake(Fake sources)
known(Known sources)
sources(SSO sources)
pointingsources(Pointing sources)
pointing(Pointing model)
candidates(Candidates)
residual(Residual map)
unknown(Blind sources)
sifted(Sifter results)
finalresult@{ shape: proc, label: "Analysis result"}

maps[(Mapcat)]
socat[(SOCAT)]
external[(External catalogs)]
canddb[(Candidate DB)]

coadd{Coadder}
simulate{Simulator}
preprocess{Preprocess}
inject{Inject}
postprocess{Postprocess}
force{Force}
point{Pointing}
find{Force}
subtract{Subtract}
blind{Blind search}
sift{Sifter}

style initial fill:#bbb,stroke:#333,stroke-width:2
style coadded fill:#bbb,stroke:#333,stroke-width:2
style processed fill:#bbb,stroke:#333,stroke-width:2
style fake fill:#bbb,stroke:#333,stroke-width:2
style known fill:#bbb,stroke:#333,stroke-width:2
style sources fill:#bbb,stroke:#333,stroke-width:2
style pointing fill:#bbb,stroke:#333,stroke-width:2
style pointingsources fill:#bbb,stroke:#333,stroke-width:2
style candidates fill:#bbb,stroke:#333,stroke-width:2
style residual fill:#bbb,stroke:#333,stroke-width:2
style sifted fill:#bbb,stroke:#333,stroke-width:2
style unknown fill:#bbb,stroke:#333,stroke-width:2
style finalresult fill:#bbb,stroke:#333,stroke-width:2
style maps fill:#4c9,stroke:#333,stroke-width:2
style socat fill:#4c9,stroke:#333,stroke-width:2
style canddb fill:#4c9,stroke:#333,stroke-width:2
style external fill:#4c9,stroke:#333,stroke-width:2
style coadd fill:#dc3,stroke:#333,stroke-width:2
style simulate fill:#dc3,stroke:#333,stroke-width:2
style preprocess fill:#dc3,stroke:#333,stroke-width:2
style inject fill:#dc3,stroke:#333,stroke-width:2
style postprocess fill:#dc3,stroke:#333,stroke-width:2
style force fill:#dc3,stroke:#333,stroke-width:2
style find fill:#dc3,stroke:#333,stroke-width:2
style point fill:#dc3,stroke:#333,stroke-width:2
style subtract fill:#dc3,stroke:#333,stroke-width:2
style blind fill:#dc3,stroke:#333,stroke-width:2
style sift fill:#dc3,stroke:#333,stroke-width:2

simulate --> fake

maps -.-> initial

initial --> coadd

coadd --> coadded

coadded --> preprocess
coadded --> find

fake --> inject

subgraph Databases

	socat
	external
	canddb
	maps

end

socat -.-> known
external -.-> sources

subgraph Analyze Map
	
	preprocess --> postprocess
	inject --> postprocess
	
	postprocess --> processed
		
	processed --> force
	known --> force
	
	force --> pointingsources
	
	pointingsources --> point
	point --> pointing
	pointing -.-> maps
		
	pointing --> find
	sources --> find
	
	find --> candidates
	
	candidates --> subtract
	
	subtract --> residual
	
	pointing --> blind
	residual --> blind
	
	blind --> unknown
	
	known --> sift
	unknown --> sift
	residual --> sift
	sources --> sift
	
	sift --> sifted

end

sifted -.-> canddb
candidates -.-> canddb
pointingsources -.-> canddb
fake -.-> canddb

sifted --> finalresult
candidates --> finalresult

```

Currently, the cross-matching follows directly in memory from the individually analyzed maps,
requires no additional inputs, and updates the records for the new candidates.

```mermaid
---
config:
  layout: elk
  look: handDrawn
---
flowchart TD

finalresult@{ shape: procs, label: "Analysis results"}
crossmatched(Cross matched results)

canddb[(Candidate DB)]

crossmatch{Cross matcher}

style finalresult fill:#bbb,stroke:#333,stroke-width:2
style crossmatched fill:#bbb,stroke:#333,stroke-width:2
style canddb fill:#4c9,stroke:#333,stroke-width:2
style crossmatch fill:#dc3,stroke:#333,stroke-width:2

finalresult --> crossmatch

crossmatch --> crossmatched

crossmatched -.-> canddb

```
