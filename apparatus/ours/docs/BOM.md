# Bill of materials and cost

Everything needed to build the apparatus, materials only (no labour, no tools beyond the drill the robot uses). Quantities come straight from the model: lumber and sheets from the cut list (`cad/tools/bom.py`), screws, bolts and hangers from the joints (`cad/fasteners.py`, plus 10 % spare), bought parts counted per item. Prices are Home Depot online prices checked 2026-10-07; your store will differ a little. Each line also has a Lowe's search for the same item. Lines marked *est.* are estimates (no Home Depot / Lowe's listing).

## Total

| Section | Cost |
|---|---:|
| Lumber | $1,537.50 |
| Sheet goods | $834.09 |
| Fasteners | $1,426.92 |
| Door | $501.89 |
| Task boards | $307.96 |
| Counter | $850.99 |
| Objects | $232.88 |
| Gantry | $155.49 |
| Finish | $105.35 |
| Electronics | $476.21 |
| **Materials total** | **$6,429.28** |
| with ~8 % sales tax | $6,943.62 |

Regenerate: `python cad/tools/bom.py && python cad/tools/shop.py`.

## Lumber

| Item | Need | Buy | Each | Cost | Links | Notes |
|---|---:|---:|---:|---:|---|---|
| 4x4 x 8 ft #2 Premium fir | 37 | 37 | $13.72 | $507.64 | [Home Depot](https://www.homedepot.com/p/4-in-x-4-in-x-8-ft-2-Premium-Grade-Fir-Dimensional-Lumber-441856/202094374) · [Lowe's](https://www.lowes.com/search?searchTerm=4x4x8+lumber) |  |
| 2x4 x 8 ft #2 Premium fir | 115 | 115 | $3.56 | $409.40 | [Home Depot](https://www.homedepot.com/p/2-in-x-4-in-x-8-ft-2-Premium-Grade-Dimensional-Lumber-441317/202094172) · [Lowe's](https://www.lowes.com/search?searchTerm=2x4x8+lumber) | $3.95 each, $3.56 at 50+ |
| 2x8 x 8 ft #2 Premium fir | 26 | 26 | $11.55 | $300.30 | [Home Depot](https://www.homedepot.com/p/2-in-x-8-in-x-8-ft-2-Premium-Grade-Fir-Dimensional-Lumber-769887252895/329444946) · [Lowe's](https://www.lowes.com/search?searchTerm=2x8x8+lumber) | $12.83 each, $11.55 at 24+ |
| 2x12 x 8 ft #2 Premium fir (stair stringers) | 4 | 4 | $24.12 | $96.48 | [Home Depot](https://www.homedepot.com/p/2-in-x-12-in-x-8-ft-2-Premium-Grade-Fir-Dimensional-Lumber-707195/202094201) · [Lowe's](https://www.lowes.com/search?searchTerm=2x12x8+lumber) |  |
| 2x10 x 16 ft #2 Premium fir (no 12 ft fir listed; cut down) | 2 | 2 | $37.47 | $74.94 | [Home Depot](https://www.homedepot.com/p/2-in-x-10-in-x-16-ft-2-Premium-Grade-Fir-Dimensional-Lumber-603724/206182037) · [Lowe's](https://www.lowes.com/search?searchTerm=2x10x12+lumber) |  |
| 2x10 x 8 ft #2 Premium fir | 3 | 3 | $18.36 | $55.08 | [Home Depot](https://www.homedepot.com/p/2-in-x-10-in-x-8-ft-2-Premium-Grade-Fir-Dimensional-Lumber-915564/206182030) · [Lowe's](https://www.lowes.com/search?searchTerm=2x10x8+lumber) |  |
| 4x4 x 10 ft #2 Premium fir | 2 | 2 | $16.92 | $33.84 | [Home Depot](https://www.homedepot.com/p/4-in-x-4-in-x-10-ft-2-Premium-Grade-Fir-Dimensional-Lumber-441864/202047663) · [Lowe's](https://www.lowes.com/search?searchTerm=4x4x10+lumber) |  |
| 2x4 x 10 ft #2 Premium fir | 3 | 3 | $8.18 | $24.54 | [Home Depot](https://www.homedepot.com/p/2-in-x-4-in-x-10-ft-2-Premium-Grade-Fir-Dimensional-Lumber-D204SEC-294-10/313824028) · [Lowe's](https://www.lowes.com/search?searchTerm=2x4x10+lumber) |  |
| 2x6 x 8 ft #2 Premium fir | 2 | 2 | $9.48 | $18.96 | [Home Depot](https://www.homedepot.com/p/2-in-x-6-in-x-8-ft-2-Premium-Grade-Fir-Dimensional-Lumber-769887254813/333311570) · [Lowe's](https://www.lowes.com/search?searchTerm=2x6x8+lumber) |  |
| 1x4 x 8 ft S4S board | 3 | 3 | $5.44 | $16.32 | [Home Depot](https://www.homedepot.com/p/1-in-x-4-in-x-8-ft-S4S-Standard-and-Construction-Premium-Kiln-Dried-Hem-Fir-Douglas-Fir-Board-Y/206517271) · [Lowe's](https://www.lowes.com/search?searchTerm=1x4x8+board) |  |

**Lumber subtotal: $1,537.50**

## Sheet goods

| Item | Need | Buy | Each | Cost | Links | Notes |
|---|---:|---:|---:|---:|---|---|
| 23/32 in (3/4 nominal) fir sheathing plywood, 4 x 8 | 17 | 17 | $42.65 | $725.05 | [Home Depot](https://www.homedepot.com/p/23-32-in-x-4-ft-x-8-ft-Fir-Sheathing-Plywood-Actual-0-688-in-x-48-in-x-96-in-439614/100034683) · [Lowe's](https://www.lowes.com/search?searchTerm=23%2F32+plywood+sheathing+4x8) | deck, ramp skins, boards |
| 7/16 in OSB sheathing, 4 x 8 | 8 | 8 | $13.63 | $109.04 | [Home Depot](https://www.homedepot.com/p/OSB-7-16-Application-as-4ft-X-8-ft-Sheathing-Panel-386081/202106230) · [Lowe's](https://www.lowes.com/search?searchTerm=7%2F16+OSB+4x8) | door wall skins, landing skirts |

**Sheet goods subtotal: $834.09**

## Fasteners

| Item | Need | Buy | Each | Cost | Links | Notes |
|---|---:|---:|---:|---:|---|---|
| Simpson ABA44Z adjustable post base, 4x4 | 16 | 16 | $14.98 | $239.68 | [Home Depot](https://www.homedepot.com/p/Simpson-Strong-Tie-ABA-ZMAX-Galvanized-Adjustable-Standoff-Post-Base-for-4x4-Nominal-Lumber-ABA44Z/100374999) · [Lowe's](https://www.lowes.com/search?searchTerm=Simpson+ABA44Z) | screwed to the deck with 6 in structural screws |
| Simpson LUS28Z joist hanger, 2x8 | 75 | 75 | $2.83 | $212.25 | [Home Depot](https://www.homedepot.com/p/Simpson-Strong-Tie-LUS-ZMAX-Galvanized-Face-Mount-Joist-Hanger-for-2x8-Nominal-Lumber-LUS28Z/100374921) · [Lowe's](https://www.lowes.com/search?searchTerm=Simpson+LUS28) | landing joists, ramp ribs |
| FastenMaster TimberLOK 6 in structural screw (50-pack) | 188 | 4 × 50-pack | $48.37 | $193.48 | [Home Depot](https://www.homedepot.com/p/FastenMaster-TimberLOK-Structural-Wood-Screws-6-inch-wood-screws-with-hex-head-Black-50-Pack-FMTLOK06-50/202268250) · [Lowe's](https://www.lowes.com/search?searchTerm=FastenMaster+TimberLOK+6+in) | 4x4 to 4x4 and end-grain joints |
| Carriage bolt 1/2 x 8 in, zinc (10-pack) | 59 | 6 × 10-pack | $26.00 | $156.00 | [Home Depot](https://www.homedepot.com/p/Everbilt-1-2-in-13-x-8-in-Zinc-Plated-Carriage-Bolt-10-Pack-800480/204281348) · [Lowe's](https://www.lowes.com/search?searchTerm=1%2F2+x+8+carriage+bolt) | through 4x4 + 4x4 or rim; zinc is fine indoors |
| Deck screw #8 x 1-5/8 in, Torx, exterior (5 lb, 735 pcs) | 2809 | 4 × 735-pack | $32.97 | $131.88 | [Home Depot](https://www.homedepot.com/p/DECKMATE-8-x-1-5-8-in-Black-Exterior-Self-Starting-Torx-Drive-Flat-Head-Deck-Screw-5-lbs-735-Piece-115922/313358324) · [Lowe's](https://www.lowes.com/search?searchTerm=1-5%2F8+in+deck+screws+5+lb) | sheet to framing, 6 in apart |
| Loctite PL 400 subfloor adhesive, 28 oz | 12 | 12 | $8.98 | $107.76 | [Home Depot](https://www.homedepot.com/p/Loctite-PL-400-Subfloor-28-oz-All-Weather-Latex-Construction-Adhesive-Tan-Cartridge-each-2136221/300751977) · [Lowe's](https://www.lowes.com/search?searchTerm=PL+400+subfloor+adhesive+28+oz) | about 1 tube per deck panel |
| Deck screw #9 x 3 in, Torx (5 lb, 365 pcs) | 961 | 3 × 365-pack | $29.97 | $89.91 | [Home Depot](https://www.homedepot.com/p/DECKMATE-9-x-3-in-Tan-Star-Flat-Head-Wood-Deck-Screw-5-lbs-Box-365-Piece-115985/306587120) · [Lowe's](https://www.lowes.com/search?searchTerm=3+in+deck+screws+5+lb) | framing joints |
| Carriage bolt 1/2 x 12 in, galvanized (each, est.) | 11 | 11 | $7.50 | $82.50 | [Home Depot](https://www.homedepot.com/s/1%2F2%20x%2012%20carriage%20bolt) · [Lowe's](https://www.lowes.com/search?searchTerm=1%2F2+x+12+carriage+bolt) | through 4x4 + doubled 2x8; price est. |
| Hex nut 1/2 in-13, galvanized (50-pack) | 70 | 2 × 50-pack | $25.98 | $51.96 | [Home Depot](https://www.homedepot.com/p/Everbilt-1-2-in-13-tpi-Galvanized-Hex-Nut-50-Pack-804070/204274106) · [Lowe's](https://www.lowes.com/search?searchTerm=1%2F2+in+galvanized+hex+nut) | one per bolt |
| Simpson 10d x 1-1/2 in HDG connector nails (600) | 749 | 2 × 600-pack | $25.21 | $50.42 | [Home Depot](https://www.homedepot.com/p/Simpson-Strong-Tie-Strong-Drive-1-1-2-in-x-0-148-in-SCN-Smooth-Shank-HDG-Connector-Nail-600-Pack-N10D5HDG-R/206101769) · [Lowe's](https://www.lowes.com/search?searchTerm=joist+hanger+nails+1-1%2F2) | 10 per hanger |
| Flat washer 1/2 in, galvanized (50-pack) | 70 | 2 × 50-pack | $22.45 | $44.90 | [Home Depot](https://www.homedepot.com/p/Everbilt-1-2-in-Galvanized-Flat-Washer-50-Pack-807300/204284546) · [Lowe's](https://www.lowes.com/search?searchTerm=1%2F2+in+galvanized+flat+washer) | one per bolt (nut side) |
| Lag screw 3/8 x 4 in, zinc (25-pack) | 32 | 2 × 25-pack | $20.98 | $41.96 | [Home Depot](https://www.homedepot.com/p/Everbilt-3-8-in-x-4-in-Hex-Zinc-Plated-Lag-Screw-25-Pack-801580/204282524) · [Lowe's](https://www.lowes.com/search?searchTerm=lag+screw+3%2F8+x+4) | post bases, keypad, counter stand |
| Simpson LSCZ adjustable stringer connector | 5 | 5 | $2.98 | $14.90 | [Home Depot](https://www.homedepot.com/p/Simpson-Strong-Tie-LSC-18-Gauge-ZMAX-Galvanized-Adjustable-Stringer-Connector-LSCZ-R/313810429) · [Lowe's](https://www.lowes.com/search?searchTerm=Simpson+LSCZ) | one per stringer |
| 3/4 in 2-hole strap (10-pack) | 8 | 1 × 10-pack | $5.98 | $5.98 | [Home Depot](https://www.homedepot.com/p/Halex-3-4-in-2-Hole-Conduit-Straps-10-Pack-26122/100165675) · [Lowe's](https://www.lowes.com/search?searchTerm=3%2F4+in+2+hole+strap) | valve nipples to blocks |
| Simpson MP36 mending plate 3 x 6 in | 2 | 2 | $1.67 | $3.34 | [Home Depot](https://www.homedepot.com/p/Simpson-Strong-Tie-MP-3-in-x-6-in-20-Gauge-Galvanized-Mending-Plate-MP36/100374920) · [Lowe's](https://www.lowes.com/search?searchTerm=mending+plate+3+x+6) |  |

**Fasteners subtotal: $1,426.92**

## Door

| Item | Need | Buy | Each | Cost | Links | Notes |
|---|---:|---:|---:|---:|---|---|
| Masonite 36 x 80 in 1-panel solid core prehung interior door | 1 | 1 | $268.00 | $268.00 | [Home Depot](https://www.homedepot.com/p/Masonite-36-in-x-80-in-1-Panel-Left-Handed-Primed-Composite-Solid-Core-Single-Prehung-Interior-Door-4-9-16-in-Flat-Jamb-83069/206580185) · [Lowe's](https://www.lowes.com/search?searchTerm=36+x+80+prehung+interior+door+solid+core) | pick the handing in Step 4 |
| Everbilt post latch gate set | 2 | 2 | $37.98 | $75.96 | [Home Depot](https://www.homedepot.com/p/Everbilt-Black-Post-Latch-Gate-Set-24417/327600104) · [Lowe's](https://www.lowes.com/search?searchTerm=gate+latch) | not in the spec; added so the gates stay shut |
| Primed MDF casing 2-1/4 in x 7 ft | 6 | 6 | $8.09 | $48.54 | [Home Depot](https://www.homedepot.com/p/366-11-16-in-x-2-1-4-in-x-84-in-Primed-Finger-Jointed-MDF-Casing-1-Piece-7-Total-Linear-Feet-10000531/202086377) · [Lowe's](https://www.lowes.com/search?searchTerm=primed+casing+7+ft) | 3 per face |
| Everbilt 10 in heavy-duty tee hinge | 4 | 4 | $10.98 | $43.92 | [Home Depot](https://www.homedepot.com/p/Everbilt-10-in-Zinc-Plated-Heavy-Duty-Standard-Tee-Hinge-24595/327600164) · [Lowe's](https://www.lowes.com/search?searchTerm=10+in+tee+hinge+heavy+duty) | 2 per gate |
| Dynasty 4 in adjustable spring hinge (2-pack) | 3 | 2 × 2-pack | $21.25 | $42.50 | [Home Depot](https://www.homedepot.com/p/Dynasty-Hardware-4-in-x-4-in-Satin-Nickel-5-8-in-Radius-Adjustable-Spring-Door-Hinge-2-Pack-SP-40-58-US15-2PK/301399076) · [Lowe's](https://www.lowes.com/search?searchTerm=4+in+spring+door+hinge) | replaces the 3 plain hinges |
| Defiant Naples passage lever | 1 | 1 | $22.97 | $22.97 | [Home Depot](https://www.homedepot.com/p/Defiant-Naples-Satin-Nickel-Hall-Closet-Door-Handle-32LYEX203B/323700934) · [Lowe's](https://www.lowes.com/search?searchTerm=passage+door+lever) |  |

**Door subtotal: $501.89**

## Task boards

| Item | Need | Buy | Each | Cost | Links | Notes |
|---|---:|---:|---:|---:|---|---|
| Everbilt 3/4 in brass full-port ball valve, FNPT | 10 | 10 | $16.78 | $167.80 | [Home Depot](https://www.homedepot.com/p/Everbilt-3-4-in-FNPT-x-3-4-in-FNPT-Lead-Free-Brass-Full-Port-Threaded-Ball-Valve-EB94ALF104/339217545) · [Lowe's](https://www.lowes.com/search?searchTerm=3%2F4+in+brass+ball+valve) |  |
| Leviton 9874 porcelain keyless lampholder | 10 | 10 | $6.25 | $62.50 | [Home Depot](https://www.homedepot.com/p/Leviton-White-660-Watt-Incandescent-Lampholder-Kit-Medium-Base-1-Piece-Single-Circuit-Keyless-Ceiling-Mount-Glazed-Porcelain-9874/301667459) · [Lowe's](https://www.lowes.com/search?searchTerm=porcelain+lampholder+keyless) | not wired; screw-in target only |
| EcoSmart A19 60 W-equivalent LED bulbs (4-pack) | 10 | 3 × 4-pack | $11.98 | $35.94 | [Home Depot](https://www.homedepot.com/p/EcoSmart-60-Watt-Equivalent-A19-Dimmable-Frosted-Glass-Filament-LED-Light-Bulb-Daylight-4-Pack-11FFA1960WESD03/318284742) · [Lowe's](https://www.lowes.com/search?searchTerm=A19+LED+bulb+60W+4+pack) | real bulbs for the screw-in task |
| 3/4 in close nipple, black steel | 10 | 10 | $1.78 | $17.80 | [Home Depot](https://www.homedepot.com/p/Southland-3-4-in-x-Close-Black-Steel-Nipple-584-001HN/100163081) · [Lowe's](https://www.lowes.com/search?searchTerm=3%2F4+close+nipple) |  |
| Rust-Oleum Painter's Touch 2X gloss green | 2 | 2 | $5.98 | $11.96 | [Home Depot](https://www.homedepot.com/p/Rust-Oleum-Painter-s-Touch-2X-12-oz-Gloss-Green-General-Purpose-Spray-Paint-334039/307244868) · [Lowe's](https://www.lowes.com/search?searchTerm=green+spray+paint) | valve crosses, cylinders |
| Rust-Oleum Painter's Touch 2X rustic orange | 2 | 2 | $5.98 | $11.96 | [Home Depot](https://www.homedepot.com/p/Rust-Oleum-Painter-s-Touch-2X-12-oz-Satin-Rustic-Orange-General-Purpose-Spray-Paint-334089/307244939) · [Lowe's](https://www.lowes.com/search?searchTerm=orange+spray+paint) | valve crosses |

**Task boards subtotal: $307.96**

## Counter

| Item | Need | Buy | Each | Cost | Links | Notes |
|---|---:|---:|---:|---:|---|---|
| Hampton Bay 30 in base cabinet, unfinished (KB30-UF) | 2 | 2 | $159.00 | $318.00 | [Home Depot](https://www.homedepot.com/p/Hampton-Bay-Recessed-Panel-30-in-W-x-24-in-D-x-34-5-in-H-Assembled-Base-Kitchen-Cabinet-in-Unfinished-KB30-UF/302970323) · [Lowe's](https://www.lowes.com/search?searchTerm=30+in+base+cabinet+unfinished) |  |
| Hampton Bay 36 in 3-drawer base, unfinished (DB36) | 1 | 1 | $238.00 | $238.00 | [Home Depot](https://www.homedepot.com/p/Hampton-Bay-36-in-W-x-24-in-D-x-34-5-in-H-Ready-to-Assemble-Drawer-Base-Kitchen-Cabinet-in-Unfinished-with-CLICKREADY-Technology-DB36/330823680) · [Lowe's](https://www.lowes.com/search?searchTerm=36+in+drawer+base+cabinet) |  |
| Hampton Bay 8 ft laminate countertop, White Ice | 1 | 1 | $159.00 | $159.00 | [Home Depot](https://www.homedepot.com/p/Hampton-Bay-8-ft-Straight-Laminate-Countertop-in-Textured-White-Ice-Granite-with-Eased-Edge-and-Integrated-Backsplash-011349010809476/310466804) · [Lowe's](https://www.lowes.com/search?searchTerm=8+ft+laminate+countertop) |  |
| Hampton Bay 30 x 24 in wall cabinet, unfinished (KW3024-UF) | 1 | 1 | $104.00 | $104.00 | [Home Depot](https://www.homedepot.com/p/Hampton-Bay-Recessed-Panel-30-in-W-x-12-in-D-x-24-in-H-Assembled-Wall-Bridge-Kitchen-Cabinet-in-Unfinished-KW3024-UF/302970248) · [Lowe's](https://www.lowes.com/search?searchTerm=30+x+24+wall+cabinet) |  |
| Franklin Brass 3 in bar pulls (25-pack) | 7 | 1 × 25-pack | $31.99 | $31.99 | [Home Depot](https://www.homedepot.com/p/Franklin-Brass-25-Pack-Solid-Bar-3-in-76-mm-Modern-Cabinet-Drawer-Pulls-in-Stainless-Steel-P15510K-SS-B/305756048) · [Lowe's](https://www.lowes.com/search?searchTerm=cabinet+pull+3+in) |  |

**Counter subtotal: $850.99**

## Objects

| Item | Need | Buy | Each | Cost | Links | Notes |
|---|---:|---:|---:|---:|---|---|
| DEWALT 20V MAX drill/driver kit DCD771C2 | 1 | 1 | $119.00 | $119.00 | [Home Depot](https://www.homedepot.com/p/DEWALT-20V-MAX-Cordless-1-2-in-Drill-Driver-2-20V-1-3Ah-Batteries-Charger-and-Bag-DCD771C2/204279858) · [Lowe's](https://www.lowes.com/search?searchTerm=DEWALT+20V+drill+kit) | the drill the robot picks up; also the build drill |
| HDX 27 gal Tough Tote, clear | 3 | 3 | $14.98 | $44.94 | [Home Depot](https://www.homedepot.com/p/HDX-27-Gal-Tough-Storage-Tote-in-Clear-with-Yellow-Lid-999-27G-C-HDX/328026917) · [Lowe's](https://www.lowes.com/search?searchTerm=clear+latching+tote+27+gal) |  |
| 16 qt milk crate | 2 | 2 | $12.98 | $25.96 | [Home Depot](https://www.homedepot.com/p/16-Qt-Polypropylene-Milk-Crate-Storage-Box-in-Black-11-in-x-13-in-x-13-in-MK131311-HD/300656648) · [Lowe's](https://www.lowes.com/search?searchTerm=milk+crate) |  |
| 3D-printer PLA 1 kg (printed bulbs and cylinders, est.) | 1 | 1 | $20.00 | $20.00 | [Home Depot](https://www.homedepot.com/s/pla%20filament) · [Lowe's](https://www.lowes.com/search?searchTerm=PLA+filament) | any 1.75 mm PLA; price est. |
| HDX 27 gal Tough Tote, black | 1 | 1 | $8.98 | $8.98 | [Home Depot](https://www.homedepot.com/p/HDX-27-Gal-Tough-Storage-Tote-in-Black-and-Yellow-999-27G-HDX/327528802) · [Lowe's](https://www.lowes.com/search?searchTerm=27+gallon+tote) |  |
| Drill target stickers (sheet, est.) | 1 | 1 | $8.00 | $8.00 | [Home Depot](https://www.homedepot.com/s/target%20stickers) · [Lowe's](https://www.lowes.com/search?searchTerm=target+stickers) | price est. |
| Small storage bin (est.) | 1 | 1 | $6.00 | $6.00 | [Home Depot](https://www.homedepot.com/s/small%20storage%20bin) · [Lowe's](https://www.lowes.com/search?searchTerm=small+storage+bin) | price est. |

**Objects subtotal: $232.88**

## Gantry

| Item | Need | Buy | Each | Cost | Links | Notes |
|---|---:|---:|---:|---:|---|---|
| Everbilt 4 in locking swivel plate caster, 250 lb | 4 | 4 | $16.97 | $67.88 | [Home Depot](https://www.homedepot.com/p/Everbilt-4-in-Red-Polyurethane-and-Steel-Swivel-Plate-Locking-Caster-Wheel-with-250-lb-Load-Rating-4120745EB/203672250) · [Lowe's](https://www.lowes.com/search?searchTerm=4+in+locking+swivel+caster) |  |
| 0.5 ton push beam trolley | 1 | 1 | $52.99 | $52.99 | [Home Depot](https://www.homedepot.com/p/SKYSHALO-Manual-Trolley-1100-lbs-0-5-Ton-Load-Capacity-Push-Beam-Trolley-with-Dual-Wheels-I-Beam-Flange-Width-2-5-to-5-5-in-FZSTXC05T2555HYFGV0-729/331971051) · [Lowe's](https://www.lowes.com/search?searchTerm=beam+trolley) | needs a steel flange; see the gantry note |
| Everbilt 1/2 in x 50 ft twisted nylon rope | 1 | 1 | $34.62 | $34.62 | [Home Depot](https://www.homedepot.com/p/Everbilt-1-2-in-x-50-ft-White-Twisted-Nylon-Rope-73272/206094359) · [Lowe's](https://www.lowes.com/search?searchTerm=1%2F2+in+nylon+rope+50+ft) | not a rated fall-arrest line |

**Gantry subtotal: $155.49**

## Finish

| Item | Need | Buy | Each | Cost | Links | Notes |
|---|---:|---:|---:|---:|---|---|
| BEHR PREMIUM ST-533 Cedar Naturaltone semi-transparent stain, 1 gal | 2 | 2 | $47.98 | $95.96 | [Home Depot](https://www.homedepot.com/p/BEHR-PREMIUM-1-gal-ST-533-Cedar-Naturaltone-Semi-Transparent-Waterproofing-Exterior-Wood-Stain-and-Sealer-553301/202263889) · [Lowe's](https://www.lowes.com/search?searchTerm=BEHR+ST-533+cedar+naturaltone+semi+transparent) | lane rails, posts, stair rails |
| Gorilla wood filler, 16 oz | 1 | 1 | $9.39 | $9.39 | [Home Depot](https://www.homedepot.com/p/Gorilla-16-oz-Natural-Wood-Filler-Tub-107103/320745459) · [Lowe's](https://www.lowes.com/search?searchTerm=wood+filler) |  |

**Finish subtotal: $105.35**

## Electronics

| Item | Need | Buy | Each | Cost | Links | Notes |
|---|---:|---:|---:|---:|---|---|
| Adafruit massive arcade button with LED, 100 mm | 25 | 25 | $9.95 | $248.75 | [Adafruit](https://www.adafruit.com/product/1185) | keypad 10, button board 10, location 5; price est. |
| Weatherproof junction box with hinged cover | 6 | 6 | $28.91 | $173.46 | [Home Depot](https://www.homedepot.com/p/NEWHOUSE-ELECTRIC-NEMA-4X-4-72-in-L-x-6-3-in-W-x-8-66-in-H-PC-Plastic-Gray-Weatherproof-Electrical-Junction-Box-with-Hinged-Cover-NHWE-M/330790682) · [Lowe's](https://www.lowes.com/search?searchTerm=electrical+project+box) | behind each button panel |
| Microcontroller board + wiring kit (est.) | 1 | 1 | $40.00 | $40.00 | [Adafruit](https://www.adafruit.com/search?q=feather) | reads the buttons and valves; price est. |
| 7-segment display + I2C backpack (est.) | 1 | 1 | $14.00 | $14.00 | [Adafruit](https://www.adafruit.com/search?q=7-segment+backpack) | keypad display; price est. |

**Electronics subtotal: $476.21**

## Cut list

Board-by-board cuts are in `build/cutlist.json` and in each step of [ASSEMBLY.md](ASSEMBLY.md). Lumber sizes are nominal: a 2x4 is 1-1/2 x 3-1/2 in, a 4x4 is 3-1/2 x 3-1/2 in, 23/32 plywood is 0.688 in thick; the model uses actual sizes.
