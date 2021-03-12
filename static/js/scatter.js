// set the dimensions and margins of the graph
var margin = {top: 20, right: 20, bottom: 30, left: 50},
    width = 1260 - margin.left - margin.right,
    height = 800 - margin.top - margin.bottom;

// parse the date / time
var parseTime = d3.timeParse("%d-%b-%y");

var allwords = []

// set the ranges
var x = d3.scaleLinear().domain([-1,1]).range([0, width]);
var y = d3.scaleLinear().domain([-1,1]).range([height, 0]);

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
          "translate(" + margin.left + "," + margin.top + ")");

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



let addText = (word) => {
  allwords.push(word);
  if(allwords.length < 2){
    svg.selectAll(".word").remove()
    let randomlocs = []
    for(let i=0; i<allwords.length; i++){
      randomlocs[i] = [Math.random(),Math.random()]
    }
    svg.selectAll("dot")
      .data(randomlocs)
      .enter()
      .attr("x", function(d) { return x(d[0]); })
      .attr("y", function(d) { return y(d[1]); })
      .attr("class","word")
      .text(function(d,i) {return allwords[i]});
  }
  else{
      d3.json('txt2pca_km',{
          method:"POST",
          body: JSON.stringify({
          //   word: word,
            allwords: allwords
          }),
          headers: {
            "Content-type": "application/json; charset=UTF-8"
          }
        })
        .then(json =>{
          console.log(json);
          let coords = json.pca;
          let clusters = json.kmeans;
          svg.selectAll(".word").remove()
          svg.selectAll("dot")
              .data(coords)
              .enter()
              .append("text")
              .attr("x", function(d) { return x(d[0]); })
              .attr("y", function(d) { return y(d[1]); })
              .attr("class","word")
              .text(function(d,i) {return allwords[i]});
          for(let i=0; i<clusters.length; i++){
            d3.select("#cluster"+(i+1))
            .text(clusters[i])
          }
    })
    .catch(e=>{
      let notfound = allwords.pop();
      console.log(e);
      alert(notfound+" was not found in the vocabulary.")
    })
  }
}


d3.select('#upload_file')
.on('click', (e)=>{
  e.preventDefault();
  var fd = new FormData();
  fd.append('image',e.target.parentElement[0].files[0]);
  fd.append('allwords', JSON.stringify(allwords));
  console.log(fd.get('allwords'));
  // console.log(e.target.parentElement[0].files[0]);
  d3.text('photo2pca',{
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
    let word = json;
    addText(word);
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