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

  // if (imageArray.length < 2)                     // if there's not a batch of images, no need to display prev and next buttons
  // {
  //   $(".prev-button").hide();
  //   $(".next-button").hide();
  // }
}

function goHome()             // add all these into 'app.js'?
{
  $(".image-results").hide();
  $(".main-content").css("display", "flex");
};

function goPrev()
{
  // if (imageIndex > 0)           // should be somewhat like this?
  // {
  //   imageIndex--;
  //   display image at index imageIndex;
  // }

  // if (imageIndex == 0)         // if there's no need for a prev button
  // {
    $(".prev-button").hide();
  //   prevHidden = true;
  // }       


  // if (nextHidden && imageArray.length > 1)              // if next is hidden and we will now need it again after going back one image
  // {
  //   $(".next-button").css("display", "flex");
  //   nextHidden = false;
  // }
};

function goNext()                 // functions are more or less the same as in goPrev, just flipped to cater for the opposite as such
{
  // if (imageIndex > imageArray.length)          
  // {
  //   imageIndex++;
  //   display image at index imageIndex;
  // }

  // if (imageIndex == (imageArray.length - 1))
  // {
    $(".next-button").hide();
  //   nextHidden = true;
  // }       


  // if (prevHidden && imageArray.length > 1)
  // {
  //   $(".prev-button").css("display", "flex");
  //   prevHidden = false;
  // }
};
