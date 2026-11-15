"""Shopping list + cost: docs/BOM.md and build/bom.json.

Quantities come from the model: lumber boards and sheets from the cut list (cad/tools/bom.py -> build/cutlist.json),
fasteners from cad/fasteners.py (plus 10 % spare), bought items counted per item. Prices and product links are Home
Depot (checked 2026-10-07, store prices vary by region); each line also has a Lowe's search link for the same item.
Electronics are not sold at either store, so those lines link to Adafruit (prices marked est.).

Run: python cad/tools/bom.py && python cad/tools/shop.py
"""
import json
import math
import os
import sys
import urllib.parse
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..')
sys.path.insert(0, os.path.join(HERE, '..'))

HD = 'https://www.homedepot.com'
CHECKED = '2026-10-07'


def lowes(q):
    return 'https://www.lowes.com/search?searchTerm=' + urllib.parse.quote_plus(q)


# key: (section, description, HD path or full url, unit price, pieces per pack, Lowe's search, note)
CAT = {
    # lumber (Douglas fir #2, kiln dried; pressure treated is not needed indoors)
    '2x4 8': ('Lumber', '2x4 x 8 ft #2 Premium fir', '/p/2-in-x-4-in-x-8-ft-2-Premium-Grade-Dimensional-Lumber-441317/202094172', 3.56, 1, '2x4x8 lumber', '$3.95 each, $3.56 at 50+'),
    '2x4 10': ('Lumber', '2x4 x 10 ft #2 Premium fir', '/p/2-in-x-4-in-x-10-ft-2-Premium-Grade-Fir-Dimensional-Lumber-D204SEC-294-10/313824028', 8.18, 1, '2x4x10 lumber', ''),
    '2x6 8': ('Lumber', '2x6 x 8 ft #2 Premium fir', '/p/2-in-x-6-in-x-8-ft-2-Premium-Grade-Fir-Dimensional-Lumber-769887254813/333311570', 9.48, 1, '2x6x8 lumber', ''),
    '2x8 8': ('Lumber', '2x8 x 8 ft #2 Premium fir', '/p/2-in-x-8-in-x-8-ft-2-Premium-Grade-Fir-Dimensional-Lumber-769887252895/329444946', 11.55, 1, '2x8x8 lumber', '$12.83 each, $11.55 at 24+'),
    '2x10 8': ('Lumber', '2x10 x 8 ft #2 Premium fir', '/p/2-in-x-10-in-x-8-ft-2-Premium-Grade-Fir-Dimensional-Lumber-915564/206182030', 18.36, 1, '2x10x8 lumber', ''),
    '2x10 12': ('Lumber', '2x10 x 16 ft #2 Premium fir (no 12 ft fir listed; cut down)', '/p/2-in-x-10-in-x-16-ft-2-Premium-Grade-Fir-Dimensional-Lumber-603724/206182037', 37.47, 1, '2x10x12 lumber', ''),
    '2x12 8': ('Lumber', '2x12 x 8 ft #2 Premium fir (stair stringers)', '/p/2-in-x-12-in-x-8-ft-2-Premium-Grade-Fir-Dimensional-Lumber-707195/202094201', 24.12, 1, '2x12x8 lumber', ''),
    '4x4 8': ('Lumber', '4x4 x 8 ft #2 Premium fir', '/p/4-in-x-4-in-x-8-ft-2-Premium-Grade-Fir-Dimensional-Lumber-441856/202094374', 13.72, 1, '4x4x8 lumber', ''),
    '4x4 10': ('Lumber', '4x4 x 10 ft #2 Premium fir', '/p/4-in-x-4-in-x-10-ft-2-Premium-Grade-Fir-Dimensional-Lumber-441864/202047663', 16.92, 1, '4x4x10 lumber', ''),
    '1x4 8': ('Lumber', '1x4 x 8 ft S4S board', '/p/1-in-x-4-in-x-8-ft-S4S-Standard-and-Construction-Premium-Kiln-Dried-Hem-Fir-Douglas-Fir-Board-Y/206517271', 5.44, 1, '1x4x8 board', ''),
    # sheets
    'ply': ('Sheet goods', '23/32 in (3/4 nominal) fir sheathing plywood, 4 x 8', '/p/23-32-in-x-4-ft-x-8-ft-Fir-Sheathing-Plywood-Actual-0-688-in-x-48-in-x-96-in-439614/100034683', 42.65, 1, '23/32 plywood sheathing 4x8', 'deck, ramp skins, boards'),
    'osb': ('Sheet goods', '7/16 in OSB sheathing, 4 x 8', '/p/OSB-7-16-Application-as-4ft-X-8-ft-Sheathing-Panel-386081/202106230', 13.63, 1, '7/16 OSB 4x8', 'door wall skins, landing skirts'),
    # fasteners and connectors
    'screw 1-5/8 in': ('Fasteners', 'Deck screw #8 x 1-5/8 in, Torx, exterior (5 lb, 735 pcs)', '/p/DECKMATE-8-x-1-5-8-in-Black-Exterior-Self-Starting-Torx-Drive-Flat-Head-Deck-Screw-5-lbs-735-Piece-115922/313358324', 32.97, 735, '1-5/8 in deck screws 5 lb', 'sheet to framing, 6 in apart'),
    'screw 3 in': ('Fasteners', 'Deck screw #9 x 3 in, Torx (5 lb, 365 pcs)', '/p/DECKMATE-9-x-3-in-Tan-Star-Flat-Head-Wood-Deck-Screw-5-lbs-Box-365-Piece-115985/306587120', 29.97, 365, '3 in deck screws 5 lb', 'framing joints'),
    'structural screw 6 in': ('Fasteners', 'FastenMaster TimberLOK 6 in structural screw (50-pack)', '/p/FastenMaster-TimberLOK-Structural-Wood-Screws-6-inch-wood-screws-with-hex-head-Black-50-Pack-FMTLOK06-50/202268250', 48.37, 50, 'FastenMaster TimberLOK 6 in', '4x4 to 4x4 and end-grain joints'),
    'carriage bolt 6': ('Fasteners', 'Carriage bolt 1/2 x 6 in, galvanized (25-pack)', '/p/Everbilt-1-2-in-13-x-6-in-Galvanized-Carriage-Bolt-25-Pack-803620/204273451', 91.48, 25, '1/2 x 6 carriage bolt galvanized', ''),
    'carriage bolt 8': ('Fasteners', 'Carriage bolt 1/2 x 8 in, zinc (10-pack)', '/p/Everbilt-1-2-in-13-x-8-in-Zinc-Plated-Carriage-Bolt-10-Pack-800480/204281348', 26.00, 10, '1/2 x 8 carriage bolt', 'through 4x4 + 4x4 or rim; zinc is fine indoors'),
    'carriage bolt 12': ('Fasteners', 'Carriage bolt 1/2 x 12 in, galvanized (each, est.)', 'https://www.homedepot.com/s/1%2F2%20x%2012%20carriage%20bolt', 7.50, 1, '1/2 x 12 carriage bolt', 'through 4x4 + doubled 2x8; price est.'),
    'hex nut': ('Fasteners', 'Hex nut 1/2 in-13, galvanized (50-pack)', '/p/Everbilt-1-2-in-13-tpi-Galvanized-Hex-Nut-50-Pack-804070/204274106', 25.98, 50, '1/2 in galvanized hex nut', 'one per bolt'),
    'washer': ('Fasteners', 'Flat washer 1/2 in, galvanized (50-pack)', '/p/Everbilt-1-2-in-Galvanized-Flat-Washer-50-Pack-807300/204284546', 22.45, 50, '1/2 in galvanized flat washer', 'one per bolt (nut side)'),
    'joist hanger': ('Fasteners', 'Simpson LUS28Z joist hanger, 2x8', '/p/Simpson-Strong-Tie-LUS-ZMAX-Galvanized-Face-Mount-Joist-Hanger-for-2x8-Nominal-Lumber-LUS28Z/100374921', 2.83, 1, 'Simpson LUS28', 'landing joists, ramp ribs'),
    'hanger nails': ('Fasteners', 'Simpson 10d x 1-1/2 in HDG connector nails (600)', '/p/Simpson-Strong-Tie-Strong-Drive-1-1-2-in-x-0-148-in-SCN-Smooth-Shank-HDG-Connector-Nail-600-Pack-N10D5HDG-R/206101769', 25.21, 600, 'joist hanger nails 1-1/2', '10 per hanger'),
    'stair stringer connector': ('Fasteners', 'Simpson LSCZ adjustable stringer connector', '/p/Simpson-Strong-Tie-LSC-18-Gauge-ZMAX-Galvanized-Adjustable-Stringer-Connector-LSCZ-R/313810429', 2.98, 1, 'Simpson LSCZ', 'one per stringer'),
    'post base 4x4': ('Fasteners', 'Simpson ABA44Z adjustable post base, 4x4', '/p/Simpson-Strong-Tie-ABA-ZMAX-Galvanized-Adjustable-Standoff-Post-Base-for-4x4-Nominal-Lumber-ABA44Z/100374999', 14.98, 1, 'Simpson ABA44Z', 'screwed to the deck with 6 in structural screws'),
    'lag screw 3/8 x 4 in': ('Fasteners', 'Lag screw 3/8 x 4 in, zinc (25-pack)', '/p/Everbilt-3-8-in-x-4-in-Hex-Zinc-Plated-Lag-Screw-25-Pack-801580/204282524', 20.98, 25, 'lag screw 3/8 x 4', 'post bases, keypad, counter stand'),
    'mending plate': ('Fasteners', 'Simpson MP36 mending plate 3 x 6 in', '/p/Simpson-Strong-Tie-MP-3-in-x-6-in-20-Gauge-Galvanized-Mending-Plate-MP36/100374920', 1.67, 1, 'mending plate 3 x 6', ''),
    'pipe strap': ('Fasteners', '3/4 in 2-hole strap (10-pack)', '/p/Halex-3-4-in-2-Hole-Conduit-Straps-10-Pack-26122/100165675', 5.98, 10, '3/4 in 2 hole strap', 'valve nipples to blocks'),
    'adhesive': ('Fasteners', 'Loctite PL 400 subfloor adhesive, 28 oz', '/p/Loctite-PL-400-Subfloor-28-oz-All-Weather-Latex-Construction-Adhesive-Tan-Cartridge-each-2136221/300751977', 8.98, 1, 'PL 400 subfloor adhesive 28 oz', 'about 1 tube per deck panel'),
    # door
    'prehung door': ('Door', 'Masonite 36 x 80 in 1-panel solid core prehung interior door', '/p/Masonite-36-in-x-80-in-1-Panel-Left-Handed-Primed-Composite-Solid-Core-Single-Prehung-Interior-Door-4-9-16-in-Flat-Jamb-83069/206580185', 268.00, 1, '36 x 80 prehung interior door solid core', 'pick the handing in Step 4'),
    'spring hinge 4 in': ('Door', 'Dynasty 4 in adjustable spring hinge (2-pack)', '/p/Dynasty-Hardware-4-in-x-4-in-Satin-Nickel-5-8-in-Radius-Adjustable-Spring-Door-Hinge-2-Pack-SP-40-58-US15-2PK/301399076', 21.25, 2, '4 in spring door hinge', 'replaces the 3 plain hinges'),
    'lever passage set': ('Door', 'Defiant Naples passage lever', '/p/Defiant-Naples-Satin-Nickel-Hall-Closet-Door-Handle-32LYEX203B/323700934', 22.97, 1, 'passage door lever', ''),
    'casing': ('Door', 'Primed MDF casing 2-1/4 in x 7 ft', '/p/366-11-16-in-x-2-1-4-in-x-84-in-Primed-Finger-Jointed-MDF-Casing-1-Piece-7-Total-Linear-Feet-10000531/202086377', 8.09, 1, 'primed casing 7 ft', '3 per face'),
    'gate hinge': ('Door', 'Everbilt 10 in heavy-duty tee hinge', '/p/Everbilt-10-in-Zinc-Plated-Heavy-Duty-Standard-Tee-Hinge-24595/327600164', 10.98, 1, '10 in tee hinge heavy duty', '2 per gate'),
    'gate latch': ('Door', 'Everbilt post latch gate set', '/p/Everbilt-Black-Post-Latch-Gate-Set-24417/327600104', 37.98, 1, 'gate latch', 'not in the spec; added so the gates stay shut'),
    # task boards
    'ball valve 3/4 in': ('Task boards', 'Everbilt 3/4 in brass full-port ball valve, FNPT', '/p/Everbilt-3-4-in-FNPT-x-3-4-in-FNPT-Lead-Free-Brass-Full-Port-Threaded-Ball-Valve-EB94ALF104/339217545', 16.78, 1, '3/4 in brass ball valve', ''),
    'nipple 3/4 x close': ('Task boards', '3/4 in close nipple, black steel', '/p/Southland-3-4-in-x-Close-Black-Steel-Nipple-584-001HN/100163081', 1.78, 1, '3/4 close nipple', ''),
    'porcelain lampholder': ('Task boards', 'Leviton 9874 porcelain keyless lampholder', '/p/Leviton-White-660-Watt-Incandescent-Lampholder-Kit-Medium-Base-1-Piece-Single-Circuit-Keyless-Ceiling-Mount-Glazed-Porcelain-9874/301667459', 6.25, 1, 'porcelain lampholder keyless', 'not wired; screw-in target only'),
    'led bulb': ('Task boards', 'EcoSmart A19 60 W-equivalent LED bulbs (4-pack)', '/p/EcoSmart-60-Watt-Equivalent-A19-Dimmable-Frosted-Glass-Filament-LED-Light-Bulb-Daylight-4-Pack-11FFA1960WESD03/318284742', 11.98, 4, 'A19 LED bulb 60W 4 pack', 'real bulbs for the screw-in task'),
    'spray green': ('Task boards', 'Rust-Oleum Painter\'s Touch 2X gloss green', '/p/Rust-Oleum-Painter-s-Touch-2X-12-oz-Gloss-Green-General-Purpose-Spray-Paint-334039/307244868', 5.98, 1, 'green spray paint', 'valve crosses, cylinders'),
    'spray orange': ('Task boards', 'Rust-Oleum Painter\'s Touch 2X rustic orange', '/p/Rust-Oleum-Painter-s-Touch-2X-12-oz-Satin-Rustic-Orange-General-Purpose-Spray-Paint-334089/307244939', 5.98, 1, 'orange spray paint', 'valve crosses'),
    # counter
    'base cabinet B30': ('Counter', 'Hampton Bay 30 in base cabinet, unfinished (KB30-UF)', '/p/Hampton-Bay-Recessed-Panel-30-in-W-x-24-in-D-x-34-5-in-H-Assembled-Base-Kitchen-Cabinet-in-Unfinished-KB30-UF/302970323', 159.00, 1, '30 in base cabinet unfinished', ''),
    'base cabinet DB36': ('Counter', 'Hampton Bay 36 in 3-drawer base, unfinished (DB36)', '/p/Hampton-Bay-36-in-W-x-24-in-D-x-34-5-in-H-Ready-to-Assemble-Drawer-Base-Kitchen-Cabinet-in-Unfinished-with-CLICKREADY-Technology-DB36/330823680', 238.00, 1, '36 in drawer base cabinet', ''),
    'wall cabinet W3024': ('Counter', 'Hampton Bay 30 x 24 in wall cabinet, unfinished (KW3024-UF)', '/p/Hampton-Bay-Recessed-Panel-30-in-W-x-12-in-D-x-24-in-H-Assembled-Wall-Bridge-Kitchen-Cabinet-in-Unfinished-KW3024-UF/302970248', 104.00, 1, '30 x 24 wall cabinet', ''),
    'countertop 8 ft': ('Counter', 'Hampton Bay 8 ft laminate countertop, White Ice', '/p/Hampton-Bay-8-ft-Straight-Laminate-Countertop-in-Textured-White-Ice-Granite-with-Eased-Edge-and-Integrated-Backsplash-011349010809476/310466804', 159.00, 1, '8 ft laminate countertop', ''),
    'cabinet pull': ('Counter', 'Franklin Brass 3 in bar pulls (25-pack)', '/p/Franklin-Brass-25-Pack-Solid-Bar-3-in-76-mm-Modern-Cabinet-Drawer-Pulls-in-Stainless-Steel-P15510K-SS-B/305756048', 31.99, 25, 'cabinet pull 3 in', ''),
    # objects
    'clear latching tote': ('Objects', 'HDX 27 gal Tough Tote, clear', '/p/HDX-27-Gal-Tough-Storage-Tote-in-Clear-with-Yellow-Lid-999-27G-C-HDX/328026917', 14.98, 1, 'clear latching tote 27 gal', ''),
    'grey tote': ('Objects', 'HDX 27 gal Tough Tote, black', '/p/HDX-27-Gal-Tough-Storage-Tote-in-Black-and-Yellow-999-27G-HDX/327528802', 8.98, 1, '27 gallon tote', ''),
    'milk crate': ('Objects', '16 qt milk crate', '/p/16-Qt-Polypropylene-Milk-Crate-Storage-Box-in-Black-11-in-x-13-in-x-13-in-MK131311-HD/300656648', 12.98, 1, 'milk crate', ''),
    'small bin': ('Objects', 'Small storage bin (est.)', 'https://www.homedepot.com/s/small%20storage%20bin', 6.00, 1, 'small storage bin', 'price est.'),
    'cordless drill': ('Objects', 'DEWALT 20V MAX drill/driver kit DCD771C2', '/p/DEWALT-20V-MAX-Cordless-1-2-in-Drill-Driver-2-20V-1-3Ah-Batteries-Charger-and-Bag-DCD771C2/204279858', 119.00, 1, 'DEWALT 20V drill kit', 'the drill the robot picks up; also the build drill'),
    'target sticker': ('Objects', 'Drill target stickers (sheet, est.)', 'https://www.homedepot.com/s/target%20stickers', 8.00, 1, 'target stickers', 'price est.'),
    'pla': ('Objects', '3D-printer PLA 1 kg (printed bulbs and cylinders, est.)', 'https://www.homedepot.com/s/pla%20filament', 20.00, 1, 'PLA filament', 'any 1.75 mm PLA; price est.'),
    # gantry
    'locking caster 4 in': ('Gantry', 'Everbilt 4 in locking swivel plate caster, 250 lb', '/p/Everbilt-4-in-Red-Polyurethane-and-Steel-Swivel-Plate-Locking-Caster-Wheel-with-250-lb-Load-Rating-4120745EB/203672250', 16.97, 1, '4 in locking swivel caster', ''),
    'beam trolley': ('Gantry', '0.5 ton push beam trolley', '/p/SKYSHALO-Manual-Trolley-1100-lbs-0-5-Ton-Load-Capacity-Push-Beam-Trolley-with-Dual-Wheels-I-Beam-Flange-Width-2-5-to-5-5-in-FZSTXC05T2555HYFGV0-729/331971051', 52.99, 1, 'beam trolley', 'needs a steel flange; see the gantry note'),
    'safety rope': ('Gantry', 'Everbilt 1/2 in x 50 ft twisted nylon rope', '/p/Everbilt-1-2-in-x-50-ft-White-Twisted-Nylon-Rope-73272/206094359', 34.62, 1, '1/2 in nylon rope 50 ft', 'not a rated fall-arrest line'),
    # finish
    'stain': ('Finish', 'BEHR PREMIUM ST-533 Cedar Naturaltone semi-transparent stain, 1 gal', '/p/BEHR-PREMIUM-1-gal-ST-533-Cedar-Naturaltone-Semi-Transparent-Waterproofing-Exterior-Wood-Stain-and-Sealer-553301/202263889', 47.98, 1, 'BEHR ST-533 cedar naturaltone semi transparent', 'lane rails, posts, stair rails'),
    'wood filler': ('Finish', 'Gorilla wood filler, 16 oz', '/p/Gorilla-16-oz-Natural-Wood-Filler-Tub-107103/320745459', 9.39, 1, 'wood filler', ''),
    # electronics (model + BOM only)
    'LED dome button 100 mm': ('Electronics', 'Adafruit massive arcade button with LED, 100 mm', 'https://www.adafruit.com/product/1185', 9.95, 1, None, 'keypad 10, button board 10, location 5; price est.'),
    '5-digit display': ('Electronics', '7-segment display + I2C backpack (est.)', 'https://www.adafruit.com/search?q=7-segment+backpack', 14.00, 1, None, 'keypad display; price est.'),
    'controller': ('Electronics', 'Microcontroller board + wiring kit (est.)', 'https://www.adafruit.com/search?q=feather', 40.00, 1, None, 'reads the buttons and valves; price est.'),
    'project box': ('Electronics', 'Weatherproof junction box with hinged cover', '/p/NEWHOUSE-ELECTRIC-NEMA-4X-4-72-in-L-x-6-3-in-W-x-8-66-in-H-PC-Plastic-Gray-Weatherproof-Electrical-Junction-Box-with-Hinged-Cover-NHWE-M/330790682', 28.91, 1, 'electrical project box', 'behind each button panel'),
}

# bought pieces in the model -> catalog key and items per model count (several pieces make one item)
ITEM_QTY = {      # explicit counts for things whose model pieces don't map 1:1 (checked against cad/model.py)
    'prehung door': 1, 'spring hinge 4 in': 3, 'lever passage set': 1, 'casing': 6, 'gate hinge': 4, 'gate latch': 2,
    'ball valve 3/4 in': 10, 'nipple 3/4 x close': 10, 'porcelain lampholder': 10, 'led bulb': 10,
    'base cabinet B30': 2, 'base cabinet DB36': 1, 'wall cabinet W3024': 1, 'countertop 8 ft': 1, 'cabinet pull': 7,
    'clear latching tote': 3, 'grey tote': 1, 'milk crate': 2, 'small bin': 1, 'cordless drill': 1,
    'target sticker': 1, 'pla': 1, 'locking caster 4 in': 4, 'beam trolley': 1, 'safety rope': 1,
    'post base 4x4': 16, 'mending plate': 2, 'pipe strap': 8, 'lag screw 3/8 x 4 in': 32,
    'LED dome button 100 mm': 25, '5-digit display': 1, 'controller': 1, 'project box': 6,
    'spray green': 2, 'spray orange': 2, 'wood filler': 1, 'stain': 2, 'adhesive': 12,
}


def quantities():
    import fasteners as F
    import model as M
    cut = json.load(open(os.path.join(ROOT, 'build', 'cutlist.json'), encoding='utf-8'))
    q = Counter()
    for nom, e in cut['lumber'].items():
        for b in e['boards']:
            q['%s %d' % (nom, b['stock'] // 12)] += 1
    for k, e in cut['sheets'].items():
        q[k] += len(e['sheets'])
    m = M.build()
    fs = F.build_fasteners(m)
    raw = Counter()
    for f in fs:
        k = f['kind']
        if k.startswith('carriage'):
            L = f['length']
            k = 'carriage bolt %d' % (6 if L <= 6.5 else (8 if L <= 8.5 else 12))
            raw['hex nut'] += 1
            raw['washer'] += 1
        raw[k] += 1
    raw['stair stringer connector'] = max(raw['stair stringer connector'], 4)   # one per stringer (4)
    raw['hanger nails'] = raw['joist hanger'] * 10
    for k, n in raw.items():
        q[k] += int(math.ceil(n * 1.1))                                        # 10 % spare
    q.update(ITEM_QTY)
    return q, Counter(f['kind'] for f in fs)


def main():
    q, fk = quantities()
    rows, missing = [], []
    for k, n in q.items():
        if k not in CAT:
            missing.append((k, n))
            continue
        sec, desc, link, price, pack, lq, note = CAT[k]
        packs = int(math.ceil(n / pack))
        rows.append(dict(key=k, section=sec, desc=desc, need=n, pack=pack, buy=packs, price=price,
                         cost=round(packs * price, 2), hd=link if link.startswith('http') else HD + link,
                         lowes=lowes(lq) if lq else '', note=note, est='est.' in desc or 'est.' in note))
    order = ['Lumber', 'Sheet goods', 'Fasteners', 'Door', 'Task boards', 'Counter', 'Objects', 'Gantry', 'Finish',
             'Electronics']
    rows.sort(key=lambda r: (order.index(r['section']), -r['cost']))
    sub = Counter()
    for r in rows:
        sub[r['section']] += r['cost']
    total = sum(sub.values())
    tax = 0.08
    out = dict(checked=CHECKED, rows=rows, subtotals=dict(sub), total=round(total, 2), tax_rate=tax,
               total_with_tax=round(total * (1 + tax), 2), missing=missing, fastener_kinds=dict(fk))
    json.dump(out, open(os.path.join(ROOT, 'build', 'bom.json'), 'w', encoding='utf-8'), indent=1)
    write_md(out, order)
    print('total $%.2f (+%.0f%% tax $%.2f)' % (total, tax * 100, total * (1 + tax)))
    for s in order:
        print('  %-12s $%9.2f' % (s, sub[s]))
    if missing:
        print('NOT PRICED:', missing)


def money(x):
    return '${:,.2f}'.format(x)


def write_md(o, order):
    L = ['# Bill of materials and cost', '',
         'Everything needed to build the apparatus, materials only (no labour, no tools beyond the drill the robot '
         'uses). Quantities come straight from the model: lumber and sheets from the cut list (`cad/tools/bom.py`), '
         'screws, bolts and hangers from the joints (`cad/fasteners.py`, plus 10 %% spare), bought parts counted per '
         'item. Prices are Home Depot online prices checked %s; your store will differ a little. Each line also has a '
         "Lowe's search for the same item. Lines marked *est.* are estimates (no Home Depot / Lowe's listing)." % o['checked'],
         '', '## Total', '',
         '| Section | Cost |', '|---|---:|']
    for s in order:
        L.append('| %s | %s |' % (s, money(o['subtotals'].get(s, 0))))
    L += ['| **Materials total** | **%s** |' % money(o['total']),
          '| with ~%d %% sales tax | %s |' % (o['tax_rate'] * 100, money(o['total_with_tax'])), '',
          'Regenerate: `python cad/tools/bom.py && python cad/tools/shop.py`.', '']
    for s in order:
        rs = [r for r in o['rows'] if r['section'] == s]
        if not rs:
            continue
        L += ['## %s' % s, '', '| Item | Need | Buy | Each | Cost | Links | Notes |', '|---|---:|---:|---:|---:|---|---|']
        for r in rs:
            buy = '%d' % r['buy'] if r['pack'] == 1 else '%d × %d-pack' % (r['buy'], r['pack'])
            links = '[Home Depot](%s)' % r['hd'] if 'homedepot' in r['hd'] else '[Adafruit](%s)' % r['hd']
            if r['lowes']:
                links += ' · [Lowe\'s](%s)' % r['lowes']
            L.append('| %s | %d | %s | %s | %s | %s | %s |' % (r['desc'], r['need'], buy, money(r['price']),
                                                               money(r['cost']), links, r['note']))
        L += ['', '**%s subtotal: %s**' % (s, money(o['subtotals'][s])), '']
    L += ['## Cut list', '', 'Board-by-board cuts are in `build/cutlist.json` and in each step of '
          '[ASSEMBLY.md](ASSEMBLY.md). Lumber sizes are nominal: a 2x4 is 1-1/2 x 3-1/2 in, a 4x4 is 3-1/2 x 3-1/2 in, '
          '23/32 plywood is 0.688 in thick; the model uses actual sizes.', '']
    open(os.path.join(ROOT, 'docs', 'BOM.md'), 'w', encoding='utf-8').write('\n'.join(L))


if __name__ == '__main__':
    main()
