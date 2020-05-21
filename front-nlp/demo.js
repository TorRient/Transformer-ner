

const url_ananlysis = "http://localhost:5000/tener"

function getColor(str) {
  switch(str){
    case "O":
      color = "#333333";
      break;
    case "B-LOC":
      color = "#4285f4";
      break;
    case "B-ORG":
      color = "#32CD32";
      break;
    case "I-LOC":
      color = "#0000CD";
      break;
    case "B-PER":
      color = "#8B0000";
      break;
    case "I-PER":
      color = "#CD5C5C";
      break;
    case "I-ORG":
      color = "#2E8B57";
      break;
    case "B-MISC":
      color = "#FFFF00";
      break;
    case "I-MISC":
      color = "#FFA500";
      break;
    case "N":
      color = "red";
      break;
    case "M":
      color = "#333333";
      break;
    }
  return color
}

function insertResult(sentence) {
  sen = '<p class="sentence">'
  sentence.forEach(ele => {
    x = ele.text
    x +=  '<span style="color:' + getColor(ele.value) + '">(' + ele.value + ')</span>'
    x += ' '
    sen += x
  })
  sen += '</p>'
  resultEle.innerHTML = sen
}

async function analysis() {
  /*
    demo
  */
  // console.log('annalysis')
  // btnSubmit.style.display = "none"
  // btnReset.style.display = "flex"
  // insertResult(sentence)


  /*
    run
  */
  const inputValue = inputEle.value
  console.log(inputValue)
  console.log(typeof(inputValue))
  try {
    const response = await axios.post(url_ananlysis, {
      text: inputValue
    });
    console.log('loading')
    if (response.status === 200) {
      let data = response.data
      let sentence = data.sentence
      insertResult(sentence)
    }
  } catch (error) {
    console.error(error);
  }
}

function reset() {
  inputEle.value = ""
  btnReset.style.display = "none"
  btnSubmit.style.display = "flex"
  resultEle.innerHTML = ""
}

const btnSubmit = document.getElementById('btn-submit')
const btnReset = document.getElementById('btn-reset')
const resultEle = document.getElementById("result")
const inputEle = document.getElementById('input')

const sentence = [
  { text: "ha noi", value: 'N' },
  { text: 'viet nam', value: 'M'}
]

btnSubmit.addEventListener('click', analysis)
btnReset.addEventListener('click', reset)


/*
  @form of data

let data = {
	sentence: [
  	{text: "hom nay", value: 'N' },
    {text: 'troi dep', value: 'M'}
  ]
}

*/