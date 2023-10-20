function sendMessage() {
    const message = document.getElementById('message').value;
    addLine("local", message)
    fetch('http://127.0.0.1:9706/query', {
        method: 'post',
        headers: {'content-type': 'application/json'},
        body: JSON.stringify({'msg': message})
    })
    .then(res => res.json())
    .then(data => {
        const audio = new Audio(data.audio);
        audio.play();
        addLine("remote", data.msg)
    });
}

function addLine(talker, msg) {
    const div = document.createElement('div');
    div.className = talker;
    div.innerHTML = msg;
    document.getElementById('messages').append(div);
}
