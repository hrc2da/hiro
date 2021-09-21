// set the dimensions and margins of the graph
var margin = {top: 20, right: 20, bottom: 30, left: 50},
    width = 1260 - margin.left - margin.right,
    height = 800 - margin.top - margin.bottom;

// parse the date / time
var parseTime = d3.timeParse("%d-%b-%y");

var allwords = []

// set the ranges
var x = d3.scaleLinear().domain([-300,300]).range([0, width*0.9]);
var y = d3.scaleLinear().domain([0,300]).range([height*0.8, 0]);

// define the line
var valueline = d3.line()
    .x(function(d) { return x(d.date); })
    .y(function(d) { return y(d.close); });

// append the svg obgect to the body of the page
// appends a 'group' element to 'svg'
// moves the 'group' element to the top left margin
var svg = d3.select("body").append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + (2*margin.left) + "," + margin.top + ")");


// DROPSHADOW CODE FROM http://bl.ocks.org/cpbotha/5200394
// filter chain comes from:
// https://github.com/wbzyl/d3-notes/blob/master/hello-drop-shadow.html
// cpbotha added explanatory comments
// read more about SVG filter effects here: http://www.w3.org/TR/SVG/filters.html

// filters go in defs element
var defs = svg.append("defs");

// create filter with id #drop-shadow
// height=130% so that the shadow is not clipped
var filter = defs.append("filter")
    .attr("id", "drop-shadow")
    .attr("height", "130%");

// SourceAlpha refers to opacity of graphic that this filter will be applied to
// convolve that with a Gaussian with standard deviation 3 and store result
// in blur
filter.append("feGaussianBlur")
    .attr("in", "SourceAlpha")
    .attr("stdDeviation", 1)
    .attr("result", "blur");

// translate output of Gaussian blur to the right and downwards with 2px
// store result in offsetBlur
filter.append("feOffset")
    .attr("in", "blur")
    .attr("dx", 1)
    .attr("dy", 1)
    .attr("result", "offsetBlur");

// overlay original SourceGraphic over translated blurred opacity by using
// feMerge filter. Order of specifying inputs is important!
var feMerge = filter.append("feMerge");

feMerge.append("feMergeNode")
    .attr("in", "offsetBlur")
feMerge.append("feMergeNode")
    .attr("in", "SourceGraphic");

// Get the data
// d3.csv("static/data.csv").then(function(data) {

//   // format the data
//   data.forEach(function(d) {
//       d.date = parseTime(d.date);
//       d.close = +d.close;
//   });

//   // Scale the range of the data
//   x.domain(d3.extent(data, function(d) { return d.date; }));
//   y.domain([0, d3.max(data, function(d) { return d.close; })]);

//   // Add the valueline path.
// //   svg.append("path")
// //       .data([data])
// //       .attr("class", "line")
// //       .attr("d", valueline);
      
//   // Add the scatterplot
//   svg.selectAll("dot")
//       .data(data)
//     .enter().append("circle")
//       .attr("r", 5)
//       .attr("cx", function(d) { return x(d.date); })
//       .attr("cy", function(d) { return y(d.close); });

//   // Add the X Axis
//   svg.append("g")
//       .attr("transform", "translate(0," + height + ")")
//       .call(d3.axisBottom(x));

//   // Add the Y Axis
//   svg.append("g")
//       .call(d3.axisLeft(y));

// });
let noteColors = ['yellow','#aff288','#a4c0ed','pink','#f0c389']
let location_dict = {};
let embedding_dict = {};
let word_order = [];
let cluster_ids = [];
let num_clusters = 5;
let c0_centers = [[-80, 130],  [-80, 190],  [0, 190],    [0, 130],    [80, 130],   [80, 190]]
let c1_centers = [[-160, 0],   [-160, 60],  [-240, 60],  [-240, 0],   [-320, 0],   [-320, 60]] 
let c2_centers = [[160, 0],    [160, 60],   [240, 60],   [240, 0],    [320, 0],    [320, 60]]
let c3_centers = [[-210, 170], [-290, 170], [-260, 230], [-180, 230], [-190, 290], [-110, 290]]
let c4_centers = [[210, 170],  [290, 170],  [260, 230],  [180, 230],  [190, 290],  [110, 290]]
let cluster_capacity = c0_centers.length
let cluster_locs = [c0_centers, c1_centers, c2_centers, c3_centers, c4_centers];
let cluster_centroids = []
for (let i=0; i<cluster_locs.length; i++){
  let cur_cluster = cluster_locs[i];
  let sumx = 0;
  let sumy = 0;
  for (let j=0; j<cur_cluster.length; j++){
    let cur_center = cur_cluster[j];
    sumx += cur_center[0];
    sumy += cur_center[1];
  }
  cluster_centroids.push([sumx/cur_cluster.length, sumy/cur_cluster.length]);
}

function wrap(text, width) {
  text.each(function() {
    let text = d3.select(this),
        words = text.text().split(/\s+/).reverse(),
        word,
        line = [],
        lineNumber = 0,
        lineHeight = 10, // px
        y = text.attr("y"),
        y_shift = 0,
        x = text.attr("x"),
        //dy = parseFloat(text.attr("dy")),
        dy = 0,
        tspan = text.text(null).append("tspan").attr("x", x).attr("y", y).attr("dy", dy);
    while (word = words.pop()) {
      line.push(word);
      tspan.text(line.join(" "));
      if (tspan.node().getComputedTextLength() > width) {
        if ((y_shift + 0) < 25){
          // console.log(y-y_start);
          y_shift += lineHeight;
          // text.attr("y",y);
          text.selectAll("tspan").attr("dy", function(d) {
            return this.getAttribute("dy")-(lineHeight/2);});
        }
        
        line.pop();
        tspan.text(line.join(" "));
        line = [word];
        tspan = text.append("tspan").attr("x", x).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy).text(word);
      }

    }
  });
}


  function dragstarted() {
    d3.select(this).attr("stroke", "blue");
  }
  function dragged(event, d) {
    d3.select(this).raise().attr("x", d.x = event.x-50).attr("y", d.y = event.y-25); //raise moves it to the top
    let textNode = d3.select(this.parentNode).select("text");
    textNode.raise().attr("x", d.x = event.x).attr("y", d.y = event.y)
    .selectAll("tspan").attr("x", d.x = event.x).attr("y", d.y = event.y)
    
    
  }

  function dragended(event, d) {
    let textNode = d3.select(this.parentNode).select("text");
    let word = textNode.attr("raw_text");
    let new_x = x.invert(event.x)
    let new_y = y.invert(event.y)
    location_dict[word] = [new_x,new_y];
    let wordIndex = word_order.indexOf(word);
    let minDistance = math.norm([cluster_centroids[0][0]-new_x, cluster_centroids[0][1]-new_y]); 
    let closestCluster = 0;
    for(let i=1; i<cluster_centroids.length; i++){
      let centroid = cluster_centroids[i];
      let distance = math.norm([centroid[0]-new_x, centroid[1]-new_y]);
      if(distance < minDistance){
        minDistance = distance;
        closestCluster = i;
      }
    }
    cluster_ids[wordIndex] = closestCluster;
    let clusterEmbeddings = [];
    let embeddingMean = [];
    for(let i=0; i<embedding_dict[word_order[0]].length; i++){
      embeddingMean.push(0);
    }
    for(let i=0; i<cluster_ids.length; i++){
      if(cluster_ids[i] == closestCluster){
        let neighborWord = word_order[i];
        let neighborEmbedding = embedding_dict[neighborWord];
        for(let j=0; j<neighborEmbedding.length; j++){
          embeddingMean[j] += neighborEmbedding[j];
        }
        clusterEmbeddings.push(neighborEmbedding);
      }
    }
    for(let i=0; i<embedding_dict[word_order[0]].length; i++){
      embeddingMean[i] /= clusterEmbeddings.length;
    }

    embedding_dict[word] = embeddingMean;
    
    d3.select(this).attr("stroke", "black")
    .attr("fill", function(d){ return noteColors[cluster_ids[word_order.indexOf(d)]];});
  }




let addText = (word) => {
  // allwords.push(word);
  // if(allwords.length < 2){
  //   svg.selectAll(".word").remove()
  //   let randomlocs = []
  //   for(let i=0; i<allwords.length; i++){
  //     randomlocs[i] = [Math.random(),Math.random()]
  //   }
  //   svg.selectAll("dot")
  //     .data(randomlocs)
  //     .enter()
  //     .attr("x", function(d) { return x(d[0]); })
  //     .attr("y", function(d) { return y(d[1]); })
  //     .attr("class","word")
  //     .text(function(d,i) {return allwords[i]});
  // }
  // else{
    d3.json('words2clusters',{
      method:"POST",
      body: JSON.stringify({
          new_word: word,
          location_dict: location_dict,
          embedding_dict: embedding_dict,
          word_order: word_order,
          cluster_ids: cluster_ids,
          num_clusters: num_clusters,
          cluster_locs: cluster_locs
    }),
          headers: {
            "Content-type": "application/json; charset=UTF-8"
          }
  }).then(json =>{
    location_dict = json.location_dict;
    embedding_dict = json.embedding_dict;
    word_order = json.word_order;
    cluster_ids = json.cluster_ids;
    num_clusters = json.num_clusters;
    cluster_locs = json.cluster_locs;

    svg.selectAll(".word").remove()
    svg.selectAll(".wordrect").remove()
    let group = svg.selectAll("dot")
      .data(word_order)
      .enter()
      .append('g')

    group
      .append('rect')
      .attr("x", function(d) {
        return x(location_dict[d][0]) - 50;
      })
      .attr("y", function(d) {
                return y(location_dict[d][1]) - 25;
            })
      .attr("fill", function(d){ return noteColors[cluster_ids[word_order.indexOf(d)]];})
      .attr("width", 100)
      .attr("height", 50)
      .attr("class", "wordrect")
      // .attr("fill", "white")
      .attr("stroke", "black")
      .style("filter", "url(#drop-shadow)")
      .call(d3.drag()
        .on("start", dragstarted)
        .on("drag", dragged)
        .on("end", dragended)
  );
  let label = group
      .append("text")
      .attr("x", function(d) {return x(location_dict[d][0]);})
      .attr("y", function(d) {return y(location_dict[d][1]);})
      .attr("raw_text", function(d) { return d;})
      .attr("text-anchor", "middle")
      .attr("class","word")
      .text(function(d) { return d;});
      // .style("font-size", function(d) { return Math.min(30, (60 - 0) / this.getComputedTextLength() * 24) + "px"; })
  })

  // svg.selectAll('.word text')
  // .call(wrap, 90)


    //   d3.json('txt2pca_km',{
    //       method:"POST",
    //       body: JSON.stringify({
    //       //   word: word,
    //         allwords: allwords
    //       }),
    //       headers: {
    //         "Content-type": "application/json; charset=UTF-8"
    //       }
    //     })
    //     .then(json =>{
    //       console.log(json);
    //       let coords = json.pca;
    //       let clusters = json.kmeans;
    //       svg.selectAll(".word").remove()
    //       svg.selectAll("dot")
    //           .data(coords)
    //           .enter()
    //           .append("text")
    //           .attr("x", function(d) { return x(d[0]); })
    //           .attr("y", function(d) { return y(d[1]); })
    //           .attr("class","word")
    //           .text(function(d,i) {return allwords[i]});
    //       for(let i=0; i<clusters.length; i++){
    //         d3.select("#cluster"+(i+1))
    //         .text(clusters[i])
    //       }
    // })
    .then(()=>{
      svg.selectAll('text')
      .call(wrap, 90);
      document.getElementById('word_input').value = '';
    })
    .catch(e=>{
      let notfound = allwords.pop();
      console.log(e);
      alert(notfound+" was not found in the vocabulary.")
    })
  
}


d3.select('#upload_file')
.on('click', (e)=>{
  e.preventDefault();
  var fd = new FormData();
  fd.append('image',e.target.parentElement[0].files[0]);
  fd.append('allwords', JSON.stringify(allwords));
  console.log(fd.get('allwords'));
  // console.log(e.target.parentElement[0].files[0]);
  d3.text('photo2emb',{
    method: "POST",
    body: fd
    // headers: {
    //   "enctype": "multipart/form-data",
    //   "Content-type": "multipart/form-data"
    // },    
    // contentType: false, 
    // processData: false, 
  })
  .then(json=>{
    console.log(json);
    // let word = json;
    // addText(word);
  })
});


d3.select('#add_text')
.on('click', (e)=>{
    e.preventDefault();
    let word = e.target.parentElement[0].value;
    addText(word);  
});



// d3.select('#add_text')
// .on('click', (e)=>{
//     e.preventDefault();
//     let word = e.target.parentElement[0].value;
//     allwords.push(word)
//     d3.json('txt2pca',{
//         method:"POST",
//         body: JSON.stringify({
//           word: word,
//           allwords: allwords
//         }),
//         headers: {
//           "Content-type": "application/json; charset=UTF-8"
//         }
//       })
//       .then(json =>{
//         console.log(json);
//         json = [json];
//         // svg.selectAll("dot")
//         //     .data(json)
//         //     .enter()
//         //     .append("circle")
//         //     .attr("r", 10)
//         //     .attr("cx", function(d) { return x(d[0]); })
//         //     .attr("cy", function(d) { return y(d[1]); })
//         //     .style("fill","blue");
//             // .text(word);
//         svg.selectAll("dot")
//             .data(json)
//             .enter()
//             .append("text")
//             // .attr("r", 10)
//             .attr("x", function(d) { return x(d[0]); })
//             .attr("y", function(d) { return y(d[1]); })
//             .text(word);
//             // .style("fill","blue");
//      })
//     }
// );
    // console.log(e.target.parentElement[0].value)})
