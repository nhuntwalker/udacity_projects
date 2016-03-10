// variables for basic chart characteristics and svg object
var svg = d3.select("#figure-container"),
    width = 960,
    height = 550,
    slide_indx = -1
    lightbox = d3.select("#lightbox-parent");


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

function draw_line_plot(all_data){
    var margin = {x: 75, top: 50, bottom: 50},
        linewidth = 2;

    d3.selectAll(".opt-box-choice.city").on("click", function(){
        d3.selectAll(".opt-box-choice.city").classed("selected", false);
        var target = d3.select(this);
            target.classed("selected", true);

        var dataset = d3.select(".opt-box-choice.subject.selected");
        if (dataset[0][0] == null){
            the_file = "pisa2012_usa_total_gender.dat"
        } else {
            the_file = dataset.attr("file-target");
        }
        new_dataset(the_file, target.attr("d-target"));
    });

    d3.selectAll(".opt-box-choice.subject").on("click", function(){
        d3.selectAll(".opt-box-choice.subject").classed("selected", false);
        var target = d3.select(this);
            target.classed("selected", true);

        var the_city = "United States of America";
        new_dataset(target.attr("file-target"), the_city);

    });

    function new_dataset(the_file, the_name){
        d3.tsv("data/" + the_file, function(d){ // slide 1
            if (d["country"] == the_name) {
                return make_numerical(d);
            }
        }, draw_line_plot);
    };
    var fem_count = all_data.filter(function(d){
                        return d.gender == "Female";
                    }).map(function(d){
                        return d["the_count"];
                    }).reduce(function(a, b){
                        return a + b;
                    }),
        male_count = all_data.filter(function(d){
                        return d.gender == "Male";
                    }).map(function(d){
                        return d["the_count"];
                    }).reduce(function(a, b){
                        return a + b;
                    }),
        popfrac_extent = d3.extent(all_data, function(d){
                    if (d.gender == "Female") {
                        return d["the_count"] / fem_count * 100;                
                    } else {
                        return d["the_count"] / male_count * 100;                
                    }
        });

    if (all_data.map(function(obj){return obj.allgrades_bucket;})[all_data.length - 1] > 1000) {
        var score_extent = [0, 3000];
    } else {
        var score_extent = [0, 1000];
    }

    var score_scale = d3.scale.linear()
                        .range([margin.x, width - margin.x])
                        .domain(score_extent),

        popfrac_scale = d3.scale.linear()
                        .range([height - margin.bottom, margin.top])
                        .domain(popfrac_extent);
                        

    var score_axis = d3.svg.axis()
                        .scale(score_scale),

        popfrac_axis = d3.svg.axis()
                        .scale(popfrac_scale)
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
            .call(score_axis);

        chart_space.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + margin.x + ", 0)")
            .call(popfrac_axis);

        chart_space.append("g")
            .attr("class", "x-label axis-label")
            .append("text")
                .attr("x", (width - margin.x)/ 2)
                .attr("y", height - margin.bottom/4)
                .text("PISA Cumulative Score");

        chart_space.append("g")
            .attr("class", "y-label axis-label")
            .style("transform", "rotate(270deg)")
            .append("text")
                .attr("x", -(height + margin.top)/2)
                .attr("y", margin.x / 3)
                .text("Population Fraction");
    } else {
        d3.selectAll(".axis").remove();
        chart_space.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + (height - margin.bottom) + ")")
            .call(score_axis);

        chart_space.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + margin.x + ", 0)")
            .call(popfrac_axis);        
    }

    // add in the data
    if (d3.select("#all-plotted-items")[0].length == 1){
        var plot_space = chart_space.append("g")
                                .attr("id", "all-plotted-items");
    } else {
        var plot_space = d3.select("#all-plotted-items")
    }

    var line = d3.svg.line()
                .interpolate("basis")
                .x(function(d) { 
                    return score_scale(d["allgrades_bucket"]); 
                })
                .y(function(d) { 
                    if (d.gender == "Female"){
                        return popfrac_scale(d["the_count"]/fem_count * 100); 
                    } else {
                        return popfrac_scale(d["the_count"]/male_count * 100); 
                    }
                });

    var datagroup = d3.nest()
        .key(function(d){
            return d.gender;
        }).entries(all_data);

    if (d3.selectAll(".country-line")[0].length == 0){
        datagroup.forEach(function(d, i) {
            plot_space.append('svg:path')
                .attr('d', line(d.values))
                .attr("class", function(){
                    if (d.key == "Female") {
                        return "country-line usa girls";
                    } else {
                        return "country-line usa boys";
                    }
                });
        });

    } else {
        d3.select(".country-line.girls")
            .transition()
            .style("opacity", 0)
            .transition()
            .attr('d', line(datagroup.filter(function(d){
                    return d.key == "Female";
                })[0]
                .values))
            .transition()
            .style("opacity", 1.0);

        d3.select(".country-line.boys")
            .transition()
            .style("opacity", 0)
            .transition()
            .attr('d', line(datagroup.filter(function(d){
                    return d.key == "Male";
                })[0]
                .values))
            .transition()
            .style("opacity", 1.0);
    } 
};

function draw_multiple_lines(all_data){
    var margin = {x: 75, top: 50, bottom: 50},
        linewidth = 2;

    var popfrac_extent = d3.extent(all_data, function(d){
            var total_count = all_data.filter(function(c){
                return c["country"] == d["country"];
            }).map(function(c){
                return c["the_count"];
            }).reduce(function(a, b){
                return a + b;
            });

            return d["the_count"] / total_count * 100;
        });

    if (all_data.map(function(obj){return obj.allgrades_bucket;})[all_data.length - 1] > 1000) {
        var score_extent = [0, 3000];
    } else {
        var score_extent = [0, 1000];
    }

    var score_scale = d3.scale.linear()
                        .range([margin.x, width - margin.x])
                        .domain(score_extent),

        popfrac_scale = d3.scale.linear()
                        .range([height - margin.bottom, margin.top])
                        .domain(popfrac_extent);
                        
    var score_axis = d3.svg.axis()
                        .scale(score_scale),

        popfrac_axis = d3.svg.axis()
                        .scale(popfrac_scale)
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
            .call(score_axis);

        chart_space.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + margin.x + ", 0)")
            .call(popfrac_axis);

        chart_space.append("g")
            .attr("class", "x-label axis-label")
            .append("text")
                .attr("x", (width - margin.x)/ 2)
                .attr("y", height - margin.bottom/4)
                .text("PISA Cumulative Score");

        chart_space.append("g")
            .attr("class", "y-label axis-label")
            .style("transform", "rotate(270deg)")
            .append("text")
                .attr("x", -(height + margin.top)/2)
                .attr("y", margin.x / 3)
                .text("Population Fraction");
    } else {
        d3.selectAll(".axis").remove();
        chart_space.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + (height - margin.bottom) + ")")
            .call(score_axis);

        chart_space.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + margin.x + ", 0)")
            .call(popfrac_axis);        
    }

    // add in the data
    if (d3.select("#all-plotted-items")[0].length == 1){
        var plot_space = chart_space.append("g")
                                .attr("id", "all-plotted-items");
    } else {
        var plot_space = d3.select("#all-plotted-items")
    }

    var line = d3.svg.line()
                .interpolate("basis")
                .x(function(d) { 
                    return score_scale(d["allgrades_bucket"]); 
                })
                .y(function(d) { 
                    var total_count = all_data.filter(function(c){
                        return c["country"] == d["country"];
                    }).map(function(c){
                        return c["the_count"];
                    }).reduce(function(a, b){
                        return a + b;
                    });

                    return popfrac_scale(d["the_count"] / total_count * 100);
                });

    var datagroup = d3.nest()
        .key(function(d){
            return d["country"];
        })
        .entries(all_data);

    function total_counts(values){
        return values.map(function(a){
            return a.the_count;
        }).reduce(function(a, b){
            return a+b;
        });
    };

    var legend = svg.append("g")
                    .attr("id", "plot-legend")
                    .attr("transform", "translate(700, " + margin.top + ")")
                        .append("text")
                        .text("Some words");

    d3.select("#plot-legend").append("text")
            .attr("id", "country-line-name");

    if (d3.selectAll(".country-line")[0].length == 0){
        datagroup.forEach(function(d, i) {
            plot_space.append('svg:path')
                .attr('d', line(d.values))
                .attr("total_counts", total_counts(d.values))
                .attr("this_country", d.key)
                .attr("class", function(){
                    if (d.key == "United States of America") {
                        return "country-line usa";
                    } else {
                        return "country-line";
                    }
                }
            );
        });
    }

    var lines = d3.selectAll(".country-line")
                .on("mouseover", function(){
                    var the_line = d3.select(this);
                    d3.select("#country-line-name")
                        .text(the_line.attr("this_country"));
                });

}

d3.json("world_countries.json", draw_map); // slide 0

// d3.tsv("data/pisa2012_usa_total_gender.dat", function(d){ // slide 1
//     if (d["country"] == "United States of America") {
//         return make_numerical(d);
//     }
// }, draw_line_plot);

function make_numerical(d){
    d["allgrades_bucket"] = +d["allgrades_bucket"];
    d["the_count"] = +d["the_count"];
    return d;
};

var options_box = d3.select("#options-box"),
    cities_line = options_box.append("div").attr("class", "options-line"),
    subject_line = options_box.append("div").attr("class", "options-line");

    cities_line.append("div")
        .attr("class", "opt-box-choice city")
        .attr("id", "usa-data")
        .attr("d-target", "United States of America")
        .html("USA total");
    cities_line.append("div")
        .attr("class", "opt-box-choice city")
        .attr("id", "florida-data")
        .attr("d-target", "Florida (USA)")
        .html("Florida");
    cities_line.append("div")
        .attr("class", "opt-box-choice city")
        .attr("id", "connecticut-data")
        .attr("d-target", "Connecticut (USA)")
        .html("Connecticut");
    cities_line.append("div")
        .attr("class", "opt-box-choice city")
        .attr("id", "massachusetts-data")
        .attr("d-target", "Massachusetts (USA)")
        .html("Massachusetts");

    subject_line.append("div")
        .attr("class", "opt-box-choice subject")
        .attr("id", "total")
        .attr("file-target", "pisa2012_usa_total_gender.dat")
        .html("Cumulative Total");
    subject_line.append("div")
        .attr("class", "opt-box-choice subject")
        .attr("id", "total")
        .attr("file-target", "pisa2012_usa_reading_gender.dat")
        .html("Reading");
    subject_line.append("div")
        .attr("class", "opt-box-choice subject")
        .attr("id", "total")
        .attr("file-target", "pisa2012_usa_science_gender.dat")
        .html("Science");
    subject_line.append("div")
        .attr("class", "opt-box-choice subject")
        .attr("id", "total")
        .attr("file-target", "pisa2012_usa_math_gender.dat")
        .html("Mathematics");

// control

var navigation = d3.selectAll(".nav").on("click", function(){
    var btn = d3.select(this),
        nugget = btn.attr("nugget");

    if (btn.attr("id") == "nav-next") {
        nugget = (Number(nugget) < 4) ? Number(nugget) + 1 : 4;
        d3.selectAll(".nav").attr("nugget", nugget)
        nav_control(nugget);

    } else {
        nugget = (Number(nugget) > 0) ? Number(nugget) - 1 : 0;
        d3.selectAll(".nav").attr("nugget", nugget)
        nav_control(nugget);
    }
});

function nav_control(advance){
    // when one of the nav buttons is clicked, either advance
    // the vis forward, or go backward
    // When moving through slides, call clear_svg()
    if (advance === 0) {
        clear_svg(svg);
        clear_html(options_box);
        show_html(lightbox);
        d3.select("#nav-prev")
            .transition()
            .style("opacity", 0)
            .transition()
            .style("display", "none");

        setTimeout(function(){
            d3.json("world_countries.json", draw_map);      
        }, 500);

    } else if (advance === 4){
        d3.select("#nav-next")
            .transition()
            .style("opacity", 0)
            .transition()
            .style("display", "none");

    } else {
        clear_svg(svg);
        clear_html(lightbox);
        d3.selectAll(".nav")
            .style("display", "block")
            .transition()
            .style("opacity", .75)

        if (advance === 1) {
            show_html(options_box);
            setTimeout(function(){
                d3.tsv("data/pisa2012_usa_total_gender.dat", function(d){ // slide 1
                    if (d["country"] == "United States of America") {
                        return make_numerical(d);
                    }
                }, draw_line_plot);
            }, 500);
  
        }
        if (advance === 2) {
            setTimeout(function(){
                d3.tsv("data/pisa2012_world_total.dat", function(d){ // slide 2
                    return make_numerical(d);
                }, draw_multiple_lines);
            }, 500);
        }
    }
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

function clear_html(object){
    object.transition()
            .duration(750)
            .style("opacity", 0);
    setTimeout(function(){
        object.style("display", "none");
    }, 750);
    return true;
}

function show_html(object){
    object.style("display", "block")
            .transition()
            .duration(1000)
            .style("opacity", 1.0);
}
/*************************
NEXT STEPS:
    add vertical line cursor event for showing the results on the Y axis
    add final map slide
    add descriptive boxes for each "slide"
*************************/