async function uploadImage() {
  submittedUI();
  const endpointURL = "https://pneumothorax.mawh.in/score"
  const apiKey = "LDzXDtJKHklAX2uhlDHRdbf5DRcTgXYf";

  const data = new FormData();
  const images = document.getElementById('customFile');
  // const images = document.querySelector('input[type="file"][multiple]');
  for (let i = 0; i < images.files.length; i++) {
    data.append('image', images.files[i]);
  }
  
  await fetch(endpointURL, {
    method: 'POST',
    body: data,
    headers: {
      'Authorization': `Bearer ${apiKey}`,
    },
  })
  .then(response => response.json())
  .then(result => {
    console.log('Success:', result);
  })
  .catch(error => {
    console.error('Error:', error);
  });
}

function submittedUI() {
  $(".main-content").hide()
  $(".image-results").css("display", "flex")

  let prevHidden = new Boolean(true);
  let nextHidden = new Boolean(false);
  let current = 0;

  // if (results.length < 2)                     // if there's not a batch of images, no need to display prev and next buttons
  // {
  //   $(".prev-button").hide();
  //   $(".next-button").hide();
  //   nextHidden = true;
  //   
  // }
}

function goHome()             // add all these into 'app.js'?
{
  $(".image-results").hide();
  $(".main-content").css("display", "flex");
};

function goPrev()               // NEED TO DECLARE VARIABLE 'current' SOMEWHERE EVERYTIME IMAGE UPLOADED (declare to 0)
{
  // if (current > 0)           // should be somewhat like this?
  // {
  //   current--;
    updateResults();
  // }

  // if (current === 0)         // if there's no need for a prev button
  // {
     document.getElementById("goPrev").disabled = true;
     prevHidden = true;
  // }
  
  // if (nextHidden && results.length > 1)              // if next is hidden and we will now need it again after going back one image
  // {
      document.getElementById("goNext").disabled = false;
      nextHidden = false;
  // }

  updateResults();
};

function goNext()                 // functions are more or less the same as in goPrev, just flipped to cater for the opposite as such
{
  // if (current < results.length)          
  // {
  //   current++;
      updateResults();
  // }

  // if (current === (results.length - 1))
  // {
    document.getElementById("goNext").disabled = true;
    nextHidden = true;
  // }       


  // if (prevHidden && results.length > 1)
  // {
    document.getElementById("goPrev").disabled = false;
     prevHidden = false;
  // }
};

function updateResults()              // changes the results being outputted currently on screen due to change in image
{
  // if (results[current].pneumothoraxDetected)               // if there is PNEUMOTHORAX detected in new image, we say that
  // {
  //    document.getElementById("detect").innerHTML = "DETECTED"; 
  //    document.getElementById("advice").innerHTML = "We recommend this image is reviewed by a medical professionnal before action is taken."
  // }

  // else
  // {
  //    document.getElementById("detect").innerHTML = "NOT DETECTED";              // if not in new image we do opposite
  //    document.getElementById("advice").innerHTML = "We don't recommend this image is reviewed by a medical professionnal at this time."
  // }

  // let accurate = results[current].confidence;
  // document.getElementById("accuracy").innerHTML = "This has been calculated with a " + accurate + " accuracy";

  // document.getElementById("image-results").src = NEW SOURCE?
};