let fileSelector = document.getElementById('customFile');
let currentFileIndex = 0;
let results = [];

function onFileInputChange(event) {
  let imageContainer = document.getElementById('image-preview-container');
  imageContainer.replaceChildren(); // ensure there are no images already being shown

  // add all images to the container
  for (let image of event.target.files) {
    let imageEl = document.createElement('img');
    imageEl.setAttribute('src', URL.createObjectURL(image));
    imageEl.classList.add('margin-border-class');
    imageEl.setAttribute('width', 100);
    imageContainer.appendChild(imageEl);
  }
}

async function onFileSubmit() {
  document.getElementById('upload-status').innerText = 'Uploading images...';
  results = await uploadImages(fileSelector.files);
  // TODO: show loading icon/text

  // switch 'screens' to the show results
  $(".main-content").css("display", "none")
  $(".image-results").css("display", "flex")

  currentFileIndex = 0;
  updateResults();

  // if there's not a batch of images, no need to display prev and next buttons
  if (results.length <= 1)
  {
    $(".prev-button").hide();
    $(".next-button").hide();
  }
}

async function uploadImages(files) {
  const endpointURL = "https://pneumothorax.mawh.in/score"
  const apiKey = "LDzXDtJKHklAX2uhlDHRdbf5DRcTgXYf";

  const data = new FormData();
  const images = document.getElementById('customFile');
  for (let i = 0; i < files.length; i++) {
    data.append(`image_${i}`, files[i]);
  }

  let fetchOptions = {
    method: 'POST',
    body: data,
    headers: {
      'Authorization': `Bearer ${apiKey}`,
    },
  };

  let response = await fetch(endpointURL, fetchOptions);
  let results = await response.json();

  return results;
}

function goHome()
{
  // TODO: get rid of jquery
  $(".image-results").hide();
  $(".main-content").css("display", "flex");

  document.getElementById('upload-status').innerText = 'Please select the images to classify';
};

function goPrev()
{
  if (currentFileIndex > 0)
  {
    currentFileIndex--;
    updateResults();
  }

  // // if there's no need for a prev button
  // if (currentFileIndex === 0)
  // {
  //    document.getElementById("goPrev").disabled = true;
  // }
  
  // // if next is hidden and we will now need it again after going back one image
  // let nextHidden = document.getElementById("goNext").disabled
  // if (nextHidden && results.length > 1)              
  // {
  //     document.getElementById("goNext").disabled = false;
  // }
};

function goNext()
{
  if (currentFileIndex < fileSelector.files.length)          
  {
    currentFileIndex++;
    updateResults();
  }

  // // if there's no need for a next button
  // if (currentFileIndex === (fileSelector.files.length - 1))
  // {
  //   document.getElementById("goNext").disabled = true;
  // }

  // // if prev is hidden and we will now need it again after going forward one image
  // let prevHidden = document.getElementById("goPrev").disabled;
  // if (prevHidden && fileSelector.files.length > 1)
  // {
  //   document.getElementById("goPrev").disabled = false;
  // }
};

// changes the results being outputted currently on screen due to change in image
function updateResults() {
  document.getElementById("goPrev").disabled = false;
  document.getElementById("goNext").disabled = false;

  if (currentFileIndex === 0) {
    document.getElementById("goPrev").disabled = true;
  }
  if (currentFileIndex === (fileSelector.files.length - 1)) {
    document.getElementById("goNext").disabled = true;
  }

  let image = fileSelector.files[currentFileIndex];
  let imageEl = document.getElementById("output");
  imageEl.src = URL.createObjectURL(image);

  let positiveCase = results[currentFileIndex].pneumothoraxDetected;
  let resultEl = document.getElementById("results-detection");
  if (positiveCase) {
     resultEl.innerText = "DETECTED";
  } else {
    resultEl.innerText = "NOT DETECTED";
  }

  let confidenceEl = document.getElementById("result-confidence");
  let confidence = (results[currentFileIndex].confidence * 100).toFixed(2);
  confidenceEl.innerText = confidence;
}
