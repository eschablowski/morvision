let fs = require('fs-extra'), path = require('path');
var contents = '';
fs.ensureDir(path.join(__dirname, 'Images'), function (err) {
    if (err)
        throw new Error(err);
    fs.readdir(path.join(__dirname, 'Images'), function (err, foldernames) {
        if (err)
            throw new Error(err);
        foldernames.forEach(function (foldername) {
            fs.readdir(path.join(__dirname, 'Images', foldername), function (err, filenames) {
                if (err)
                    throw new Error(err);
                filenames.forEach(function (filename) {
                    contents += foldername + ',' + filename+ '\n';
                    fs.writeFile(path.join(__dirname, 'annotationCache', 'DirLists', 'labelme.txt'), contents);
                });
            });
        });
    });
});