const allNotes = ['', 'c', 'c#', 'd', 'd#', 'e', 'f', 
    'f#', 'g', 'g#', 'a', 'a#', 'b'];
// div containing all keys
const keys = document.getElementById('keys');

const generate = key => {
    const keyElement = '<div class="key">' +
        '<div class="bar" id="' + key + '"></div>' +
        '<div class="key-name">' + key + '</div>' +
        '</div>';
    return keyElement;
}

// generate and add all keys 
keys.innerHTML = allNotes.reduce((accum, key) => accum + generate(key));