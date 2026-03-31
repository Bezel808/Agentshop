# UI Regression Notes

## Baseline
- Search page switched from dense grid cards to Amazon-like list results with left filter rail.
- Product detail switched to three-column structure: gallery / info / buybox.

## Validation checklist
- Desktop (>=1200px): sidebar visible, result list readable, buybox on right.
- Tablet (900-1199px): one-column catalog layout, detail sections stacked.
- Mobile (<640px): search header stacked, result item and buybox full width.

## Known intentional differences vs Amazon
- No Amazon logo or brand assets.
- No pixel-level typography cloning.
- Preserved existing backend data schema and routes.
