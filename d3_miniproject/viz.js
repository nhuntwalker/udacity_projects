// Setup the plot
var width = 960,
    height = 420,
    bar_width = 100,
    dx = 10,
    x_round = 10,
    y_round = 5,
    ymax = 1550000,
    repo_href = "https://raw.githubusercontent.com/nhuntwalker/udacity_projects/master/d3_miniproject/";

// Add the plot
var title = d3.select("body").append("h2").text("Top Brands by Social Media Presence"),
    svg = d3.select("body").append("svg");
svg.attr("width", width).attr("height", height);

var xscale = d3.scale.linear().range([0, width]).domain([0, 8]),
    yscale = d3.scale.linear().range([0, height]).domain([0, ymax]),
    the_data = [{brand: "YouTube", count: 1452921, logo: repo_href + "youtube-logo.png"},
                {brand: "eBay", count: 1158424, logo: repo_href + "ebay-logo.png"},
                {brand: "Google", count: 780929, logo: repo_href + "google-logo.png"},
                {brand: "Apple", count: 517532, logo: repo_href + "apple-logo.png"},
                {brand: "Samsung", count: 374786, logo: repo_href + "samsung-logo.png"},
                {brand: "Skype", count: 374786, logo: repo_href + "skype-logo.png"},
                {brand: "Disney", count: 270867, logo: repo_href + "disney-logo.png"},
                {brand: "Nokia", count: 259728, logo: repo_href + "nokia-logo.png"}],
    bars = svg.selectAll("rect")
                .data(the_data).enter()
                .append("rect"),
    bar_attrs = bars
                    .attr("x", function(d, ii){return ii*(bar_width + dx) + dx/2;})
                    .attr("y", function(d){return height - yscale(d.count);})
                    .attr("width", bar_width)
                    .attr("height", function(d){return yscale(d.count);})
                    .attr("class", "data-bar")
                    .attr("rx", x_round)
                    .attr("ry", y_round)
                    .text(function(d){return d.brand;});

var logo_height = 30,
    images = svg.selectAll("image")
                .data(the_data).enter()
                .append("image"),
    image_attrs = images
                    .attr("xlink:href", function(d){return d.logo})
                    .attr("height", logo_height)
                    .attr("width", bar_width-20)
                    .attr("x", function(d, ii){return ii*(bar_width + dx) + 10 + dx/2;})
                    .attr("y", height - logo_height - 5);

var texts = svg.selectAll("text")
                .data(the_data).enter()
                .append("text"),
    text_attrs = texts
                    .attr("height", logo_height)
                    .attr("width", bar_width-20)
                    .attr("x", function(d, ii){return ii*(bar_width + dx) + 10 + dx/2;})
                    .attr("y", function(d){return height - (yscale(d.count) - 20);})
                    .text(function(d){return d.count.toLocaleString(0);})
                    .attr("dx", function(d){
                        if (d.count >= 1E6) {
                            return 2.5;
                        } else {
                            return 10;
                        }
                    })
                    .attr("class", "data-text");

var yscale2 = d3.scale.linear().range([height, 0])
                .domain([0, ymax/1000]);

var y_axis = d3.svg.axis()
                .scale(yscale2).orient("left")
                .ticks(10).tickSize(5);

var axis_bar = svg.append("rect").attr("height", height + 10)
                    .attr("width", 80).attr("class", "sidebar")
                    .attr("x", 888).attr("y", 0)
                    .attr("fill", "gray");

svg.append("svg:g")
    .attr("transform", 'translate(930, 0)')
    .call(y_axis)
    .style({"fill":"none","stroke":"black","stroke-width":"1"});

svg.append("text").attr("id", "y-title")
    .attr("transform", "rotate(270)")
    .attr("x", -height/2).attr("text-anchor", "middle")
    .attr("y", width - 10)
    .text("Media Presence Count (in thousands)");
