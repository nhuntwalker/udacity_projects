// variables for basic chart characteristics and svg object
var svg = d3.select("#figure-container"),
    width = 960,
    height = 550,
    slide_indx = -1;

function nav_control(advance){
    // when one of the nav buttons is clicked, either advance
    // the vis forward, or go backward, but not back to the
    // title slide
    // When moving through slides, call clear_svg()
}

function clear_svg(parent){
    // when called, fades then removes all child elements 
    // from the input element
    parent.selectAll("*")
        .transition()
        .duration(500)
        .style("opacity", 0)
        .remove();

    return true;
}

function draw_map(geo_data){
    "use strict";
    // setup the projection
    var projection = d3.geo.mercator()
                        .scale(150)
                        .translate([width/2, height/1.5]);

    var path = d3.geo.path().projection(projection);

    var map = svg.selectAll("path")
                .data(geo_data.features)
                .enter()
                .append("path")
                .attr("d", path)
                .attr("class", "map-country");

    function place_points(location_data){
        
        // drop points for selection onto map
        var points = svg.append("g")
                        .attr("class", "drop-points")
                        .selectAll("circle")
                        .data(location_data)
                        .enter()
                        .append("circle")
                        .attr("cx", function(d){
                            var coords = projection([+d.longitude, +d.latitude]);
                            return coords[0];
                        })
                        .attr("cy", function(d){
                            var coords = projection([+d.longitude, +d.latitude]);
                            return coords[1];
                        })
                        .attr("r", 5)
                        .attr("class", "location-points");
    }

    d3.json("locations.JSON", place_points);
};

function draw_box_plot(all_data){
    var margin = {x: 75, top: 100, bottom: 50},
        boxwidth = 40;

    var grade_scale = d3.scale.linear()
                        .range([margin.x, width - margin.x])
                        .domain([7, 12]),
        score_scale = d3.scale.linear()
                        .range([height - margin.bottom, margin.top])
                        .domain([0, 3000]);

    var grade_axis = d3.svg.axis()
                        .scale(grade_scale)
                        .tickValues([7,8,9,10,11,12]),
        score_axis = d3.svg.axis()
                        .scale(score_scale)
                        .orient("left");

    if (d3.select("#full-chart")[0].length == 1){
        var chart_space = svg.append("g")
                            .attr("id", "full-chart");   
    } else {
        var chart_space = d3.select("#full-chart");
    }

    if (d3.selectAll(".axis")[0].length == 0) {
        chart_space.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + (height - margin.bottom) + ")")
            .call(grade_axis);

        chart_space.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + margin.x + ", 0)")
            .call(score_axis);

        chart_space.append("g")
            .attr("class", "x-label axis-label")
            .append("text")
                .attr("x", (width - margin.x)/ 2)
                .attr("y", height - margin.bottom/4)
                .text("Grade Level");

        chart_space.append("g")
            .attr("class", "y-label axis-label")
            .style("transform", "rotate(270deg)")
            .append("text")
                .attr("x", -(height + margin.top)/2)
                .attr("y", margin.x / 3)
                .text("PISA Score");
    }

    // add in the data
    if (d3.select("#all-plotted-items")[0].length == 1){
        var plot_space = chart_space.append("g")
                                .attr("id", "all-plotted-items");
    } else {
        var plot_space = d3.select("#all-plotted-items")
    }

    plot_space.selectAll("rect")
        .data(all_data)
        .enter()
        .append("rect")
        .attr("class", function(d){
            if (d["gender"] == "Female") {
                return "plotted-box female";
            } else if (d["gender"] == "Male") {
                return "plotted-box male";
            } else {
                return "plotted-box neutral";
            }
        })
        .attr("x", function(d){
            if (d["gender"] == "Female") {
                return grade_scale(d["grade_level"]) - boxwidth/2;
            } else if (d["gender"] == "Male") {
                return grade_scale(d["grade_level"]) + boxwidth/2;
            } else {
                return grade_scale(d["grade_level"]) - boxwidth/2;
            }
        })
        .attr("width", function(d){
            if (d["gender"]) {
                return boxwidth;
            } else {
                return boxwidth*2;
            }
        })
        .attr("y", function(d){
            return score_scale(d["total_avg"] + d["total_std"]);
        })
        .attr("height", function(d){
            return -(score_scale(d["total_avg"] + d["total_std"]) - score_scale(d["total_avg"] - d["total_std"]));
        });

};

function draw_line_plot(all_data){
    var margin = {x: 75, top: 100, bottom: 50},
        linewidth = 2;

    var score_extent = d3.extent(all_data, function(d){
        return d["total_avg"]
    })

    var grade_scale = d3.scale.linear()
                        .range([margin.x, width - margin.x])
                        .domain([7, 12]),
        score_scale = d3.scale.linear()
                        .range([height - margin.bottom, margin.top])
                        .domain(score_extent);
                        

    var grade_axis = d3.svg.axis()
                        .scale(grade_scale)
                        .tickValues([7,8,9,10,11,12]),
        score_axis = d3.svg.axis()
                        .scale(score_scale)
                        .orient("left");

    if (d3.select("#full-chart")[0].length == 1){
        var chart_space = svg.append("g")
                            .attr("id", "full-chart");   
    } else {
        var chart_space = d3.select("#full-chart");
    }

    if (d3.selectAll(".axis")[0].length == 0) {
        chart_space.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + (height - margin.bottom) + ")")
            .call(grade_axis);

        chart_space.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + margin.x + ", 0)")
            .call(score_axis);

        chart_space.append("g")
            .attr("class", "x-label axis-label")
            .append("text")
                .attr("x", (width - margin.x)/ 2)
                .attr("y", height - margin.bottom/4)
                .text("Grade Level");

        chart_space.append("g")
            .attr("class", "y-label axis-label")
            .style("transform", "rotate(270deg)")
            .append("text")
                .attr("x", -(height + margin.top)/2)
                .attr("y", margin.x / 3)
                .text("PISA Score");
    }

    // add in the data
    if (d3.select("#all-plotted-items")[0].length == 1){
        var plot_space = chart_space.append("g")
                                .attr("id", "all-plotted-items");
    } else {
        var plot_space = d3.select("#all-plotted-items")
    }

    var line = d3.svg.line()
                // .interpolate("basis")
                .x(function(d) { 
                    return grade_scale(d["grade_level"]); 
                })
                .y(function(d) { 
                    return score_scale(d["total_avg"]); 
                });

    var datagroup = d3.nest()
        .key(function(d){
            return d.country;
        }).entries(all_data);

    datagroup.forEach(function(d, i) {
        plot_space.append('svg:path')
            .attr('d', line(d.values))
            .attr("class", function(){
                if (d.key == "United States of America") {
                    return "country-line usa";
                } else {
                    return "country-line";
                }
            });
    });
 
};

// d3.json("world_countries.json", draw_map); // slide 0
 
// d3.tsv("pisa2012_usa_gender.dat", function(d){ // slide 1
//     if (d["country"] == "United States of America") {
//         return make_numerical(d);
//     }
// }, draw_box_plot);

// d3.tsv("pisa2012_usa_total.dat", function(d){ // slide 2
//     return make_numerical(d);
// }, draw_line_plot);

d3.tsv("pisa2012_world_total.dat", function(d){ // slide 3
    return make_numerical(d);
}, draw_line_plot);

function make_numerical(d){
    d["grade_level"] = +d["grade_level"];
    d["math_avg"] = +d["math_avg"];
    d["read_avg"] = +d["read_avg"];
    d["scie_avg"] = +d["scie_avg"];
    d["total_avg"] = +d["total_avg"];

    d["math_min"] = +d["math_min"];
    d["read_min"] = +d["read_min"];
    d["scie_min"] = +d["scie_min"];
    d["total_min"] = +d["total_min"];

    d["math_max"] = +d["math_max"];
    d["read_max"] = +d["read_max"];
    d["scie_max"] = +d["scie_max"];
    d["total_max"] = +d["total_max"];

    d["math_std"] = +d["math_std"];
    d["read_std"] = +d["read_std"];
    d["scie_std"] = +d["scie_std"];
    d["total_std"] = +d["total_std"];
    return d;
};

var total_btn = d3.select("#gender-total"),
    gender_btn = d3.select("#gender-split");

total_btn.on("click", function(){
    d3.select("#all-plotted-items").remove();
    d3.tsv("pisa2012_usa_total.dat", function(d){
    if (d["country"] == "United States of America") {
        return make_numerical(d);
    }
}, draw_box_plot)});

gender_btn.on("click", function(){
    d3.select("#all-plotted-items").remove();
    d3.tsv("pisa2012_usa_gender.dat", function(d){
    if (d["country"] == "United States of America") {
        return make_numerical(d);
    }
}, draw_box_plot)});

/*************************
NEXT STEPS:
    bar chart slide ranking countries
    add final map slide
    add descriptive boxes for each "slide"
*************************/