var fs = require('fs'), archiver = require('archiver'), path = require('path');

var output = fs.createWriteStream(__dirname + '/data.zip');
var archive = archiver('zip');

output.on('close', function() {
  console.log(archive.pointer() + ' total bytes');
  console.log('archiver has been finalized and the output file descriptor has closed.');
});

archive.on('error', function(err) {
  throw err;
});

archive.pipe(output);

['Annotations', 'Masks', 'Images', 'Scribbles', 'data'].forEach(function (folder) {
    archive.directory(path.join(__dirname, folder), folder);
});

archive.finalize();

archive.on('progress', function (obj) {
    process.stdout.write('\r\x1b[K');
    process.stdout.write(obj.entries.processed + ' out of ' + obj.entries.total + ' compressed');
});