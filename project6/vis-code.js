// variables for basic chart characteristics and svg object
var svg = d3.select("#figure-container"),
    width = 960,
    height = 600;


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
                        .style("fill", "black");
    }

    d3.json("locations.JSON", place_points);
}

d3.json("world_countries.json", draw_map);