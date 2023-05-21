from pricelist2 import pricelist2

pricelist = {

  "Board and Batten": sum(pricelist2.get('Board and Batten')),
  "Masonry Siding": sum(pricelist2.get('Masonry Siding')),
  "Lap Siding": sum(pricelist2.get('Lap Siding')),
  "Shingled Siding": sum(pricelist2.get('Shingled Siding')),
  "Stone Siding": sum(pricelist2.get('Stone Siding')),
  "Stucco Siding": sum(pricelist2.get('Stucco Siding')),
  "Log Siding":sum(pricelist2.get('Log Siding')),
  "House Wrap": sum(pricelist2.get("House Wrap")),


  "Laminated Shingles": sum(pricelist2.get('Laminated Shingles')),
  "3-Tab Shingles": sum(pricelist2.get('3-Tab Shingles')),
  "Roof Tile Demo": sum(pricelist2.get('Roof Tile Demo')),
  "S-Shaped Clay Tiles": sum(pricelist2.get('S-Shaped Clay Tiles')),
  "Built Up Roofing": sum(pricelist2.get('Built Up Roofing')),
  "Shingle Removal": sum(pricelist2.get("Shingle Removal")),
  "Metal Roofing": sum(pricelist2.get('Metal Roofing')),
  "Metal Roofing Removal": sum(pricelist2.get('Metal Roofing Removal')),
  "Metal Ridge Cap": sum(pricelist2.get('Metal Ridge Cap')),
  "Roof Felt": sum(pricelist2.get('Roof Felt')),
  "Built Up Removal": sum(pricelist2.get('Built Up Removal')),
  "Flat Tile": sum(pricelist2.get('Flat Tile')),
  "Ridge Cap": sum(pricelist2.get('Ridge Cap')),
  "Tile Ridge Cap": sum(pricelist2.get('Tile Ridge Cap')),
  "Metal Ridge Cap": sum(pricelist2.get('Metal Ridge Cap')),
  "Roof Vent Flashing": sum(pricelist2.get('Roof Vent Flashing')),
  "Drip Edge": sum(pricelist2.get("Drip Edge")),
  "Detach and Reset Solar Panel": sum(pricelist2.get("Detach and Reset Solar Panel")),

  "Steep Charge": sum(pricelist2.get('Steep Charge')),
  "Very Steep Charge": sum(pricelist2.get("Very Steep Charge")),
  "High Charge": sum(pricelist2.get('High Charge')),

  "Valley Metal": sum(pricelist2.get('Valley Metal')),
  "Vented Ridge Cap": sum(pricelist2.get('Vented Ridge Cap')),
  "Chimney Flashing": sum(pricelist2.get('Chimney Flashing')),
  "Detach Satellite Dish": sum(pricelist2.get("Detach Satellite Dish")),
  "Skylight Flashing": sum(pricelist2.get('Skylight Flashing')),

  "Drywall": sum(pricelist2.get('Drywall')),
  "Paint": sum(pricelist2.get('Paint')),
  "Baseboards": sum(pricelist2.get("Baseboards")),
  "Paint Baseboards": sum(pricelist2.get("Paint Baseboards")),
  "Insulation": sum(pricelist2.get("Insulation")),

  "Board Fence": sum(pricelist2.get("Board Fence")),
  "Chain Link Fence": sum(pricelist2.get("Chain Link Fence")),
  "Wrought Iron Fence": sum(pricelist2.get("Wrought Iron Fence")),
  "Picket Fence": sum(pricelist2.get("Picket Fence")),
  "Rail Fence": sum(pricelist2.get("Rail Fence")),
  "Paint Fence": sum(pricelist2.get("Paint Fence")),

  "Carpet": sum(pricelist2.get("Carpet")),
  "Vinyl": sum(pricelist2.get("Vinyl")),
  "Engineered": sum(pricelist2.get("Engineered")),
  "Laminated": sum(pricelist2.get("Laminated")),
  "Reducer Strip": sum(pricelist2.get('Reducer Strip')),

  "Debris Haul": sum(pricelist2.get("Debris Haul")),
  "Dumpster": sum(pricelist2.get("Dumpster")),
}

print(pricelist)
