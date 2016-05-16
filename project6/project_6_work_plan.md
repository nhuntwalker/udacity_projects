### Project 6 Work Plan

- Decide on narrative:
	- The performance of the United States in Math, Reading, Science, and overall with respect to the rest of the world
	- Show split by grade
	- Show further split by gender
	- Open up at the end to investigate individual countries/regions
- ~~Upload **all** data to MySQL database~~
- ~~Extract relevant columns:~~

| Column Name | Full Name | Column Number |
| ----------- | --------- | ------------- |
| CNT | Country | 0 |
|ST01Q01 | International Grade | 7 |
|ST03Q01 | Birth - Month | 9 |
|ST03Q02 | Birth -Year | 10 |
|ST04Q01 | Gender | 11 |
|ST06Q01 | Age at ISCED 1 | 13 |
|WEALTH | Wealth | 487 |
|PV1MATH | Plausible Value 1 in Math | 500 |
|PV2MATH | ... | 500 |
|PV3MATH | ... | 501 |
|PV4MATH | ... | 502 |
|PV5MATH | ... | 503 |
|PV1READ | Plausible Value 1 in Reading | 504 |
|PV2READ | ... | 541 |
|PV3READ | ... | 542 |
|PV4READ | ... | 543 |
|PV5READ | ... | 544 |
|PV1SCIE | Plausible Value 1 in Science | 545 |
|PV2SCIE | ... | 546 |
|PV3SCIE | ... | 547 |
|PV4SCIE | ... | 548 |
|PV5SCIE | ... | 549 |

- ~~Note: removing Birth month and birth year because somewhat redundant with age. Need economy of data.~~
- ~~Create new columns: PVMATHAVG, PVREADAVG, PVSCIEAVG~~
- ~~Get latitudes and longitudes of each country~~
- ~~Create lat-lon JSON data set~~
- ~~Add country regions to lat-lon data set (e.g. North America, Central America, Caribbean, Oceania, etc.; source: https://en.wikipedia.org/wiki/United_Nations_geoscheme)~~
- ~~Create world map, coloring countries that are present in data~~
- ~~Add secondary dashboard for seeing overall distributions w.r.t. grade, split by gender~~
- ~~Add tooltips for countries showing data~~

### Project 6 Remaining Work

#### Slide 2:

- ~~Change x-axis label when changing axis data~~
- ~~Change y-axis label to indicate that it's the percentage of the population~~
- Put the label of the overall data set up in the top left of the graph
- ~~Add an explanatory box in the top right of the chart epxlaining the lines and their colors, as well as how to work the graph using the controls at the bottom.~~
- See about having the axis and data controls show up pre-selected
- Consider adding a tool-tip that lets you see the actual population count for each line for a given PISA score.

#### Slide 3: 

- Add explanatory box in the top-left part of the graph.
- Put a border on the top right box
- ~~Add axis labels~~
- ~~Make x-axis label change with changing data~~
- Remove "Selected Country" line and change the next line down to "Selected Country"
- Investigate "Science" and "Mathematics" data. They may be the same.

#### Slide 4:

- Add explanatory box in the top left
- Add legend in the bottom right

#### Slide 5:

- Add explanatory box in the top left

#### General:

- Make all legend boxes and explanatory boxes black with white text.

