// variables for basic chart characteristics and svg object
var svg = d3.select("#figure-container"),
    width = 960,
    height = 550,
    slide_indx = -1,
    lightbox = d3.select("#lightbox-parent"),
    tooltip = d3.select("body")
                .append("div")
                .attr("id", "tooltip")
                .style("top", (height - 154) + "px")
                .style("left", 10 + "px");

tooltip.append("div")
        .attr("id", "tooltip-country")
        .attr("class", "tooltip-div");

tooltip.append("div")
        .attr("id", "tooltip-flag")
        .attr("class", "tooltip-div")
        .append("img");

tooltip.append("div")
        .attr("class", "clearfix");

tooltip.append("div")
        .attr("id", "tooltip-students")
        .attr("class", "tooltip-div")
        .append("div")
        .html("Students: <span></span>");

tooltip.append("div")
        .attr("id", "tooltip-region")
        .attr("class", "tooltip-div")
        .append("div")
        .html("<span id='region'></span> Rank:<br/><i class='fa fa-mars'></i>: <span id='region-rank-male'></span> <span class='gender-sep'>|</span> <i class='fa fa-venus'></i>: <span id='region-rank-female'></span>");

tooltip.append("div")
        .attr("class", "tooltip-div")
        .attr("id", "tooltip-worldwide")
        .append("span")
        .html("Worldwide Rank:");

tooltip.append("div")
        .attr("id", "tooltip-rankings-male")
        .attr("class", "col-6")
        .html('<i class="fa fa-mars"></i><ul class="rank-list"><li id="male-overall"></li> <li id="male-science"></li> <li id="male-reading"></li> <li id="male-math"></li></ul>');

tooltip.append("div")
        .attr("id", "tooltip-rankings-female")
        .attr("class", "col-6")
        .html('<i class="fa fa-venus"></i><ul class="rank-list"><li id="female-overall"></li> <li id="female-science"></li> <li id="female-reading"></li> <li id="female-math"></li></ul>');


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
// debugger;
    function place_points(location_data){
        // actual academic information

        function populate_tooltip(academic_data){
            // drop points for selection onto map

            function rank_these_on_this(input_data, column){
                // take in some objects and sort them in order based on the
                // input column
                return input_data.sort(function(a,b){
                    return b[column] - a[column];
                }).map(function(d, ii){
                    d["rank"] = ii + 1;
                    return d;
                });
            }

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
                            .attr("r", 4)
                            .attr("region-name", function(d){
                                return d["region"];
                            })
                            .attr("class", "location-points")
                            .on("mouseover", function(d){
                                d3.selectAll(".location-points")
                                    .attr("r", 4)
                                    .classed("selected", false)
                                    .classed("in-region", false);

                               d3.selectAll(d3.selectAll(".location-points")[0]
                                    .filter(function(loc){
                                        return d3.select(loc)
                                                    .attr("region-name") == d["region"];
                                    }))
                                    .attr("r", 6)
                                    .classed("in-region", true);

                                d3.select(this)
                                    .classed("selected", true);

                                var country_data = academic_data.filter(function(scores){
                                        return scores["country"] == d["country"];
                                    }),
                                    female = country_data[0],
                                    male = country_data[1];

                                tooltip.style("display", "block")
                                        .transition()
                                        .duration(500)
                                        .style("opacity", 0.9);

                                d3.select("#tooltip-country")
                                    .html(d["country"]);

                                d3.select("#tooltip-students span")
                                    .html((female["the_count"] + male["the_count"])
                                    .toLocaleString());

                                var places_in_region = location_data.filter(function(loc){
                                        return loc["region"] === d["region"];
                                    }).map(function(loc){
                                        return loc["country"];
                                    }),
                                    male_data_in_region = academic_data.filter(function(scores){
                                        return (places_in_region.indexOf(scores["country"]) > -1) && (scores["gender"] == "Male");
                                    }),
                                    female_data_in_region = academic_data.filter(function(scores){
                                        return (places_in_region.indexOf(scores["country"]) > -1) && (scores["gender"] == "Female");
                                    }),
                                    male_ranking = rank_these_on_this(male_data_in_region, "overall_avg"),
                                    female_ranking = rank_these_on_this(female_data_in_region, "overall_avg");

                                d3.select("#tooltip-region #region")
                                    .html(d["region"]);

                                d3.select("#tooltip-region #region-rank-male")
                                    .html(male_ranking
                                    .filter(function(loc){
                                        return loc["country"] == d["country"];
                                    })
                                    .map(function(about_time){
                                        return about_time["rank"];
                                    }) + " / " + male_ranking.length);

                                d3.select("#tooltip-region #region-rank-female")
                                    .html(female_ranking.filter(function(loc){
                                        return loc["country"] == d["country"];
                                }).map(function(about_time){
                                    return about_time["rank"];
                                }) + " / " + female_ranking.length);

                                function get_ranking_only(objects_to_rank, column, country){
                                    // do all the rank processing and return ONLY a string with the result
                                    var ranked_objects = rank_these_on_this(objects_to_rank, column);
                                    var the_rank = ranked_objects.filter(function(loc){
                                                return loc["country"] == country;
                                            }).map(function(cnt_data){
                                                return cnt_data["rank"];
                                            });
                                    return the_rank + " / " + ranked_objects.length;
                                }

                                d3.select("#tooltip-flag img")
                                    .attr('src', get_flag(d["country"]));

                                d3.select("#male-math")
                                    .html("Math: " + get_ranking_only(academic_data.filter(function(item){
                                        return item["gender"] == "Male";
                                    }), "math_avg", d["country"]));

                               d3.select("#male-science")
                                    .html("Science: " + get_ranking_only(academic_data.filter(function(item){
                                        return item["gender"] == "Male";
                                    }), "scie_avg", d["country"]));

                                d3.select("#male-reading")
                                    .html("Reading: " + get_ranking_only(academic_data.filter(function(item){
                                        return item["gender"] == "Male";
                                    }), "read_avg", d["country"]));

                                d3.select("#male-overall")
                                    .html("Overall: " + get_ranking_only(academic_data.filter(function(item){
                                        return item["gender"] == "Male";
                                    }), "overall_avg", d["country"]));

                                d3.select("#female-math")
                                    .html("Math: " + get_ranking_only(academic_data.filter(function(item){
                                        return item["gender"] == "Female";
                                    }), "math_avg", d["country"]));

                               d3.select("#female-science")
                                    .html("Science: " + get_ranking_only(academic_data.filter(function(item){
                                        return item["gender"] == "Female";
                                    }), "scie_avg", d["country"]));

                                d3.select("#female-reading")
                                    .html("Reading: " + get_ranking_only(academic_data.filter(function(item){
                                        return item["gender"] == "Female";
                                    }), "read_avg", d["country"]));

                                d3.select("#female-overall")
                                    .html("Overall: " + get_ranking_only(academic_data.filter(function(item){
                                        return item["gender"] == "Female";
                                    }), "overall_avg", d["country"]));
                            });

        }
        d3.tsv("pisa2012_world_averages_gender.dat", function(d){
            return make_numerical(d);
        }, populate_tooltip);

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
            the_file = dataset.attr("file-target-1");
        }
        new_dataset(the_file, target.attr("d-target"));
    });

    d3.selectAll(".opt-box-choice.subject").on("click", function(){
        d3.selectAll(".opt-box-choice.subject").classed("selected", false);
        var target = d3.select(this);
            target.classed("selected", true);

        var the_city = "United States of America";
        new_dataset(target.attr("file-target-1"), the_city);

    });

    function new_dataset(the_file, the_name){
        d3.tsv("" + the_file, function(d){ // slide 1
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

    d3.selectAll(".opt-box-choice.subject").on("click", function(){
        d3.selectAll(".opt-box-choice.subject").classed("selected", false);
        var target = d3.select(this);
            target.classed("selected", true);

        new_dataset(target.attr("file-target-2"));

    });
    function new_dataset(the_file){
        d3.tsv("" + the_file, function(d){ // slide 1
                return make_numerical(d);
        }, draw_multiple_lines);
    };

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
                    .attr("transform", "translate(700, " + margin.top + ")");

    if (d3.select("#legend-head")[0][0] === null) {
        legend.append("text")
            .attr("id", "legend-head")
            .text("Selected Country");        
    }

    legend.append("text")
            .attr("y", 15)
            .attr("id", "country-line-name");

    legend.append("text")
            .attr("id", "population-line")
            .attr("y", 30)

    legend.append("image")
        .attr("id", "flag-line")
        .attr("y", 45);

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
            // For filling the legend
    var lines = d3.selectAll(".country-line")
                .on("mouseover", function(){
                    var the_line = d3.select(this);
                    d3.selectAll(".country-line").classed("green", false);
                    the_line.classed("green", true);
                    d3.select("#country-line-name")
                        .text("Country: " + the_line.attr("this_country"));

                    d3.select("#population-line")
                        .text("Number of students: " + Number(the_line.attr("total_counts")).toLocaleString());

                    d3.select("image#flag-line")
                        .attr('xlink:href', get_flag(the_line.attr("this_country")));

                    if (d3.select("#flag-border")[0][0] === null){
                        legend.append("rect")
                            .attr("id", "flag-border")
                            .attr("y", 45)
                            .attr("width", 68)
                            .attr("height", 40);
                    }
                });
    } else {
        clear_svg(d3.selectAll("path.country-line"));
        setTimeout(function(){
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

                // For filling the legend
    var lines = d3.selectAll(".country-line")
                .on("mouseover", function(){
                    var the_line = d3.select(this);
                    d3.selectAll(".country-line").classed("green", false);
                    the_line.classed("green", true);
                    d3.select("#country-line-name")
                        .text("Country: " + the_line.attr("this_country"));

                    d3.select("#population-line")
                        .text("Number of students: " + Number(the_line.attr("total_counts")).toLocaleString());

                    d3.select("image#flag-line")
                        .attr('xlink:href', get_flag(the_line.attr("this_country")));

                    if (d3.select("#flag-border")[0][0] === null){
                        legend.append("rect")
                            .attr("id", "flag-border")
                            .attr("y", 45)
                            .attr("width", 68)
                            .attr("height", 40);
                    }
                });
        }, 750);
    }
};

function draw_scatter_plots(all_data){
    var margin = {x: 75, top: 50, bottom: 50},
        linewidth = 2;

    d3.selectAll(".opt-box-choice.axis-data").on("click", function(){
        var target = d3.select(this);
        if (target.classed("xval")) {
            d3.selectAll(".xval").classed("selected", false);
        } else {
            d3.selectAll(".yval").classed("selected", false);            
        }
        target.classed("selected", true);

        new_dataset(target.attr("d-target"));

    });
    function new_dataset(){
        d3.tsv("pisa2012_world_averages.dat", function(d){ // slide 1
                return make_numerical(d);
        }, draw_scatter_plots);
    };

    var score_extent = [350, 625];

    var x_scale = d3.scale.linear()
                        .range([margin.x, width - margin.x])
                        .domain(score_extent),

        y_scale = d3.scale.linear()
                        .range([height - margin.bottom, margin.top])
                        .domain(score_extent);
                        
    var x_axis = d3.svg.axis()
                        .scale(x_scale),

        y_axis = d3.svg.axis()
                        .scale(y_scale)
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
            .call(x_axis);

        chart_space.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + margin.x + ", 0)")
            .call(y_axis);
    } else {
        d3.selectAll(".axis").remove();
        chart_space.append("g")
            .attr("class", "x axis")
            .attr("transform", "translate(0," + (height - margin.bottom) + ")")
            .call(x_axis);

        chart_space.append("g")
            .attr("class", "y axis")
            .attr("transform", "translate(" + margin.x + ", 0)")
            .call(y_axis);
    }

    if (d3.selectAll(".axis-label")[0].length == 0) {
        chart_space.append("g")
            .attr("class", "x-label axis-label")
            .append("text")
                .attr("x", (width - margin.x)/ 2)
                .attr("y", height - margin.bottom/4)
                .text("PISA Score - Reading");

        chart_space.append("g")
            .attr("class", "y-label axis-label")
            .style("transform", "rotate(270deg)")
            .append("text")
                .attr("x", -(height + margin.top)/2)
                .attr("y", margin.x / 3)
                .text("PISA Score - Math");
    } else {
        var new_xlabel = d3.selectAll(".xval.selected").html(),
            new_ylabel = d3.selectAll(".yval.selected").html();

        d3.select(".x-label.axis-label text").text("PISA Score - " + new_xlabel);       
        d3.select(".y-label.axis-label text").text("PISA Score - " + new_ylabel);      
    }

    // get the data
    if (d3.select("#all-plotted-items")[0].length == 1){
        var plot_space = chart_space.append("g")
                                .attr("id", "all-plotted-items");
    } else {
        var plot_space = d3.select("#all-plotted-items")
    }

    if (d3.selectAll(".data-points")[0][0] == null) {
        var points = plot_space.selectAll("circle")
                .data(all_data)
                .enter()
                .append("circle")
                .attr("class", function(d){
                    if (d["country"] == "United States of America") {
                        return "data-points usa";
                    } else {
                        return "data-points";
                    }
                })
                .attr("cx", function(d){return x_scale(d[d3.select("#x-axis-choices .selected").attr("d-target")]);})
                .attr("cy", function(d){return y_scale(d[d3.select("#y-axis-choices .selected").attr("d-target")]);})
                .attr("r", 6);

        // point functionality
        points.on("mouseover", function(){
            d3.selectAll(".data-points").classed("selected", false);
            d3.select(this).classed("selected", true);
        });
    } else {
        var points = d3.selectAll(".data-points")
                .transition()
                .delay(function(d,ii){
                    return ii*10;
                })
                .attr("cx", function(d){return x_scale(d[d3.select("#x-axis-choices .selected").attr("d-target")]);})
                .attr("cy", function(d){return y_scale(d[d3.select("#y-axis-choices .selected").attr("d-target")]);})
                .attr("r", 6);
    }

    // add a one-to-one line
    var line_data = [
                    {xval: score_extent[0], yval: score_extent[0]}, 
                    {xval: score_extent[1], yval: score_extent[1]}
                    ];

    var line = d3.svg.line()
        .x(function(d) {
            return x_scale(d.xval);
        })
        .y(function(d) {
            return y_scale(d.yval);
        })
        .interpolate('basis');

    plot_space.append('svg:path')
            .attr("d", line(line_data))
            .attr('stroke', "#CCC") // line color
            .attr('stroke-width', 2) // line width
            .attr('class','diag_line guide')
            .attr('fill', 'none');  
};

d3.json("world_countries.json", draw_map); // slide 0/4

// d3.tsv("pisa2012_usa_total_gender.dat", function(d){ // slide 1
//     if (d["country"] == "United States of America") {
//         return make_numerical(d);
//     }
// }, draw_line_plot);

// d3.tsv("pisa2012_world_averages.dat", function(d){ // slide 3
//     return make_numerical(d);
// }, draw_scatter_plots);

function make_numerical(d){
    d["allgrades_bucket"] = +d["allgrades_bucket"];
    d["the_count"] = +d["the_count"];
    d["math_avg"] = +d["math_avg"];
    d["scie_avg"] = +d["scie_avg"];
    d["read_avg"] = +d["read_avg"];
    d["math_std"] = +d["math_std"];
    d["scie_std"] = +d["scie_std"];
    d["read_std"] = +d["read_std"];

    d["overall_avg"] = d["math_avg"] + d["scie_avg"] + d["read_avg"];
    return d;
};

// setting up the options box
var options_box = d3.select("#options-box"),
    cities_line = options_box.append("div")
                                .attr("class", "options-line"),
    subject_line = options_box.append("div")
                                .attr("class", "options-line"),
    x_axis_options = options_box.append("div")
                                .attr("class", "options-line")
                                .attr("id", "x-axis-choices"),
    y_axis_options = options_box.append("div")
                                .attr("class", "options-line")
                                .attr("id", "y-axis-choices");

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
        .attr("file-target-1", "pisa2012_usa_total_gender.dat")
        .attr("file-target-2", "pisa2012_world_total.dat")
        .html("Cumulative Total");

    subject_line.append("div")
        .attr("class", "opt-box-choice subject")
        .attr("id", "total")
        .attr("file-target-1", "pisa2012_usa_reading_gender.dat")
        .attr("file-target-2", "pisa2012_world_reading.dat")
        .html("Reading");

    subject_line.append("div")
        .attr("class", "opt-box-choice subject")
        .attr("id", "total")
        .attr("file-target-1", "pisa2012_usa_science_gender.dat")
        .attr("file-target-2", "pisa2012_world_science.dat")
        .html("Science");

    subject_line.append("div")
        .attr("class", "opt-box-choice subject")
        .attr("id", "total")
        .attr("file-target-1", "pisa2012_usa_science_gender.dat")
        .attr("file-target-2", "pisa2012_world_science.dat")
        .html("Mathematics");

    x_axis_options.append("div")
        .attr("id", "x-label-control")
        .attr("class", "opt-box-label")
        .html("X-axis Data");

    x_axis_options.append("div")
        .attr("class", "opt-box-choice axis-data xval")
        .attr("d-target", "math_avg")
        .html("Math");

    x_axis_options.append("div")
        .attr("class", "opt-box-choice axis-data xval")
        .attr("d-target", "scie_avg")
        .html("Science");

    x_axis_options.append("div")
        .attr("class", "opt-box-choice axis-data xval selected")
        .attr("d-target", "read_avg")
        .html("Reading");

    y_axis_options.append("div")
        .attr("id", "y-label-control")
        .attr("class", "opt-box-label")
        .html("Y-axis Data");

    y_axis_options.append("div")
        .attr("class", "opt-box-choice axis-data yval selected")
        .attr("d-target", "math_avg")
        .html("Math");

    y_axis_options.append("div")
        .attr("class", "opt-box-choice axis-data yval")
        .attr("d-target", "scie_avg")
        .html("Science");

    y_axis_options.append("div")
        .attr("class", "opt-box-choice axis-data yval")
        .attr("d-target", "read_avg")
        .html("Reading");

// utility

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
    // When moving through slides, call clear_svg_children()
    if (advance === 0) {
        clear_svg_children(svg);
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
        clear_svg_children(svg);
        clear_html(options_box);
        enable_html(tooltip);
        d3.select("#nav-next")
            .transition()
            .style("opacity", 0)
            .transition()
            .style("display", "none");

        setTimeout(function(){
            d3.json("world_countries.json", draw_map);              
        }, 600);

    } else {
        clear_svg_children(svg);
        clear_html(lightbox);
        reset_options();
        d3.selectAll(".nav")
            .style("display", "block")
            .transition()
            .style("opacity", .75)

        if (advance === 1) {
            clear_html(x_axis_options);
            clear_html(y_axis_options);
            disable_html(tooltip);
            show_html(options_box);
            show_html(d3.select(".options-line"));
            setTimeout(function(){
                d3.tsv("pisa2012_usa_total_gender.dat", function(d){ // slide 1
                    if (d["country"] == "United States of America") {
                        return make_numerical(d);
                    }
                }, draw_line_plot);
            }, 500);
  
        }
        if (advance === 2) {
            clear_html(d3.selectAll(".options-line"));
            setTimeout(function(){
                show_html(subject_line);
                d3.tsv("pisa2012_world_total.dat", function(d){ // slide 2
                    return make_numerical(d);
                }, draw_multiple_lines);
            }, 500);
        }
        if (advance === 3) {
            disable_html(tooltip);
            clear_html(d3.selectAll(".options-line"));
            setTimeout(function(){
                show_html(x_axis_options);
                show_html(y_axis_options);

                var x_set = d3.selectAll("#x-axis-choices .axis-data")
                                .classed("selected", function(d, ii){
                                    if (ii == 2) {
                                        return true;
                                    } else {
                                        return false;
                                    }
                                }),
                    y_set = d3.selectAll("#y-axis-choices .axis-data")
                                .classed("selected", function(d, ii){
                                    if (ii == 0) {
                                        return true;
                                    } else {
                                        return false;
                                    }
                                });

                d3.tsv("pisa2012_world_averages.dat", function(d){ // slide 3
                    return make_numerical(d);
                }, draw_scatter_plots);
            }, 500);
        }
    }
}

function clear_svg_children(parent){
    // when called, fades then removes all child elements 
    // from the input element
    parent.selectAll("*")
        .transition()
        .duration(500)
        .style("opacity", 0)
        .remove();

    return true;
}

function clear_svg(object){
    object.transition().duration(300).style("opacity", 0).remove();
    return true;
}

function clear_html(object){
    object.transition()
            .duration(750)
            .style("opacity", 0);
    setTimeout(function(){
        object.style("display", "none");
    }, 500);
    return true;
}

function disable_html(object){
    object.style("display", "none");
    return true;
}
function enable_html(object){
    setTimeout(function(){
        object.style("display", "block");        
    }, 500);
}

function show_html(object){
    object.style("display", "block")
            .transition()
            .duration(1000)
            .style("opacity", 1.0);
}

function reset_options(){
    d3.selectAll(".opt-box-choice").classed("selected", false);
}

function get_flag(country_name){
    var mod_name = country_name.replace(/[\s\(]+/g, "-").replace(/[\)]+/g, "");
    return "https://github.com/nhuntwalker/udacity_projects/blob/master/project6/flags/flag-of-" + mod_name + ".png";
}
