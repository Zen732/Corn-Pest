# Photo Collection Guide (Corn)

## What to photograph

| Class | Photograph |
|-------|-----------|
| Fall armyworm | Caterpillars in the whorl; damaged whorl leaves with ragged holes and sawdust-like droppings |
| Corn planthopper | Small hoppers on leaves and in the whorl; black sooty mould; drying leaves |
| Corn aphid | Colonies on leaves and tassels; black sooty leaf surface |
| Armyworm | Caterpillars; leaf edges eaten, plants stripped to the midrib |
| Common cutworm | Caterpillars; young plants cut or eaten bare |
| Healthy | Green corn leaves, whorls and stalks with no damage |
| Unidentified damage | Damaged or diseased corn leaves with no insect in view |

Photograph what the app user will photograph: corn plants in the field. Leaves, whorls and
stalks work best; corn ears are not supported by the severity measurement yet.

## How to shoot

- Fill most of the frame with the plant. Soil, mud and straw in the photo count as damage.
- Vary lighting (morning, midday, shade), angle and distance.
- Keep the subject in focus.
- One problem per photo; avoid photos with several different pests.
- Green, growing plants: dry, mature leaves read as damage.
- JPG or PNG, a phone camera is fine.

## Keep test photos separate

- Put photos taken in Cabadbaran corn fields in `ml/dataset/test_local/corn/<class>/`.
- Never copy them into `raw/`. Training on them would make the final test meaningless.

## Check your progress

```
python ml/src/count_images.py              # training images (ml/dataset/raw)
python ml/src/count_images.py test_local   # Cabadbaran test images
```
