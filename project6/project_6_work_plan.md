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
- Create world map, coloring countries that are present in data
- Make country color dependent on overall plausible value quantile
- Add secondary dashboard for seeing overall distributions w.r.t. grade, split by gender
- Add tooltips for countries showing data
