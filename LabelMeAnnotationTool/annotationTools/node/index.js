var fs = require('fs-extra');
var express = require('express');
var path = require('path');
var bodyParser = require('body-parser');
var archiver = require('archiver');
var parseXML = require('xml-parser');
var toXML = require('object-to-xml');
var fg = require('fast-glob');
let lwip = require('lwip');


var app = express();
app.use(bodyParser.json());       // to support JSON-encoded bodies
app.use(bodyParser.urlencoded({     // to support URL-encoded bodies
    extended: true,
    limit: 10 * 1024 * 1024
}));
// app.use(bodyParser.raw({
//     type: '*/*'
// }));
app.use(require('express-xml-bodyparser')());

app.use(function (req, res, next) {
    console.log('connected to %s with %s.', req.path, req.method);
    next();
})

app.post('/annotationTools/php/createdir.php', function (req, res) {
    var dirName = path.normalize(path.join(__dirname, '..', '..', req.body.urlData));
    fs.ensureDir(dirName).then(function () {
        res.sendStatus(200);
    }).catch(function (err) {
        res.sendStatus(500);
    });
});

app.post('/annotationTools/php/encode.php', function (req, res) {
    var width = parseInt(req.body.width);
    var height = parseInt(req.body.height);
    var framerate = parseFloat(req.body.rate);
    var inpath = path.normalize(path.join(__dirname, '..', '..', req.body.input)) | './';
    var initframe = parseFloat(req.body.frame);
    var duration = parseFloat(req.body.duration);

    var foldername = path.parse(inpath).dir;

    var files = [];
    var count = 0;
    var format = '%010d';
    var mode = true;
    fs.readdir(inpath).then(function (file) {
        if (count == 1) {
            var entry = file;
            if (file.indexOf('_') !== -1) {
                mode = false;
                entry = entry.substring(entry.indexOf('_') + 1);
            }
            var length = entry.length - 4;
            format = '%0' + length + 'd'
        }
        if (file.substring(-4).toLowerCase() == '.jpg') count++;
    });
    count--;
    var lastFrame = initframe + duration * framerate;
    if (lastFrame >= count) lastFrame = count;
    var i = initframe;
    for (; i <= lastFrame; i++) {
        var file;
        if (mode == 0) file = inpath + '/' + foldername + '_' + sprintf(format, i) + '.jpg';
        else file = inpath + '/' + sprintf(format, i) + '.jpg';
        files.push(file);
    }
    var output = "{\r\n";
    output += "frm:\"JSVID\",\r\n";
    output += "ver:" + version + ",\r\n";
    output += "width:" + width + ",\r\n";
    output += "height:" + height + ",\r\n";
    output += "rate:" + framerate + ",\r\n";
    output += "firstframe:" + initframe + ",\r\n";
    output += "frames:" + count + ",\r\n";
    output += "data:{\r\n";
    output += "video:[\r\n";
    for (var i = 0; i < files.length; i++) {
        var size = fs.statSync(files[i]).size;
        var enc = fs.readFileSync(files[i]).toString('base64');
        output += "\"data:image/jpeg;base64," + enc + "\"";
        if (i < files.length - 1) output += ",";
        output += "\r\n";
    }
    output += "]\r\n";
    output += "}\r\n";
    output += "}\r\n";
    res.end(output);
});

app.post('/annotationTools/php/getpackfile.php', function (req, res) {
    var archive = archiver('zip');

    var folder = req.body.folder;
    var imgname = req.body.name;
    var collection = path.basename(folder);
    var zipname = collection + '_' + imgname.substring(0, imgname.length - 4) + '.zip';
    var imgurl = toolHome('Images', folder, imgname);
    var xmlname = imgname.substring(0, imgname.length - 4) + '.xml';
    var xmlurl = toolHome('Annotations', folder, xmlname);
    console.log(xmlname);

    archive.on('error', function (err) {
        res.status(500).send({ error: err.message });
    });

    //on stream closed we can end the request
    archive.on('end', function () {
        console.log('Archive wrote %d bytes', archive.pointer());
    });

    //set the archive name
    res.attachment(zipname);

    //this is the streaming magic
    archive.pipe(res);

    archive.file(imgurl, { name: imgname });
    archive.file(xmlurl, { name: xmlname });
    var masks = fg.sync(path.relative(process.cwd(), toolHome('Masks', folder, imgname.substring(0, imgname.length - 4) + '_mask_*')));

    archive.glob(path.relative(process.cwd(), toolHome('Masks', folder, imgname.substring(0, imgname.length - 4) + '_mask_*')));
    archive.glob(path.relative(process.cwd(), toolHome('Scribbles', folder, imgname.substring(0, imgname.length - 4) + '_scribble_*')));

    archive.finalize();
    // var archive = archiver('zip', {
    //     zlib: { level: 9 } // Sets the compression level.
    // });
    // archive.pipe(res);
    // var folder = req.body.folder;

    // var imgname = req.body.name;
    // var collection = path.basename(folder);
    // var zipname = collection + '_' + imgname.substring(0, -4) + '.zip';
    // var imgurl = toolHome('Images', folder, imgname);
    // var xmlname = collection + '_' + imgname.substring(0, -4) + '.xml';
    // var xmlurl = toolHome('Annotations', folder, xmlname);

    // archive.file(imgurl, { name: imgname });
    // archive.file(xmlurl, { name: xmlname });

    // archive.glob(toolHome('Masks', folder, imgname.substring(0, -4), '_mask_*'));
    // archive.glob(toolHome('Scribbles', folder, imgname.substring(0, -4), '_scribble_*'));

    // archive.on('progress', function (obj) {
    //     if (obj.entries.total == obj.entries.processed) {
    //         res.header('Content-Description', 'File Transfer');
    //         res.header('Content-type', 'application/zip');
    //         res.header('Content-Type', 'application/force-download');
    //         res.header('Content-Disposition', 'attachment; filename=' + zipname);
    //         res.header('Expires', '0');
    //         res.header('Cache-Control', 'must-revalidate, post-check=0, pre-check=0');
    //         res.header('Pragma', 'public');
    //         res.header('Content-Length', obj.fs.totalBytes);
    //     }
    // });
});

app.post('/annotationTools/php/saveimage.php', function (req, res) {
    var data = Buffer.from(req.body.image.split(',', 2)[1], 'base64');
    var file = toolHome(req.body.uploadDir, req.body.name);
    fs.writeFile(file, data).then(function (filename) {
        res.send(filename);
    }).catch(function (err) {
        res.status(500).send('Unable to save this image.');
        console.log(err);
    });
});

app.get('/annotationTools/perl/fetch_image.cgi', function (req, res) {
    var mode = req.query.mode;
    var username = req.query.username;
    var collection = req.query.collection;
    var folder = req.query.folder;
    var image = req.query.image;
    var imDir;
    var imFile;
    switch (mode) {
        case 'mt':
        case 'i':
            var fname = toolHome('annotationCache', 'DirLists', collection + '.txt');
            if (!fs.existsSync(fname)) {
                res.sendStatus(404);
                //return;

            }
            fs.readFile(fname).then(function (val) {
                var tmp = val.toString();
                tmp = tmp.split('\n');
                if (image == void 0 || image == null) {
                    tmp = tmp.map(function (val) {
                        return val.split(',', 2);
                    });
                    tmp.pop();
                    tmp.forEach(function (element, index) {
                        if (typeof image != 'number' && element[1] == image)
                            image = index;
                    });
                    console.log(tmp);
                    console.log(image);
                    tmp = tmp[image + 1 >= tmp.length ? 0 : image + 1];
                }
                else
                    tmp = tmp[Math.round(Math.random() * (tmp.length - 1))].split(',');
                res.setHeader('Content-type', 'text/xml');
                res.send('<out><dir>' + tmp[0] + '</dir><file>' + tmp[1] + '</file></out>');
                res.end();
            }).catch(function (err) {
                res.sendStatus(500);
                console.log(err);
            });
            break;
        case 'c':
            var fname = toolHome('annotationCache', 'DirLists', collection + '.txt');
            if (!fs.existsSync(fname)) {
                res.sendStatus(404);
                return;
            }
            fs.readFile(fname).then(function (val) {
                var tmp = val.toString();
                var images = [];
                var folders = [];
                var temp = tmp.split('\n');
                for (var i = 0; i < temp.length; temp++) {
                    var val = temp[i].split(',');
                    images.push(val[1]);
                    folders.push(val[0]);
                }
                res.send('<out><dir>' + folders[image + 1] + '</dir><file>' + images[image + 1] + '</file></out>');
                res.end();
            }).catch(function () {
                res.sendStatus(500);
            });
            break;
        case 'f':
            var fname = toolHome('Images', folder);
            fs.readdir(fname).then(function (images) {
                var regex = RegExp(image + '$');
                images.filter(function (val) {
                    return val.match(regex);
                });
                res.send('<out><dir>' + folder + '</dir><file>' + images[0] + '</file></out>');
            }).catch(function () {
                res.sendStatus(404);
            });
            break;
    }
});
app.get('/annotationTools/perl/fetch_prev_image.cgi', function (req, res) {
    var mode = req.params.mode;
    var username = req.params.username;
    var collection = req.params.collection;
    var folder = req.params.folder;
    var image = req.params.image;
    var imDir;
    var imFile;
    switch (mode) {
        case 'mt':
        case 'i':
            var fname = toolHome('annotationCache', 'DirLists', collection + '.txt');
            if (!fs.existsSync(fname)) {
                res.sendStatus(404);
                return;
            }
            fs.readFile(fname).then(function (val) {
                var tmp = val.toString();
                tmp = tmp.split('\n').pop();
                tmp = tmp.split(',');
                res.setHeader('Content-type', 'text/xml');
                res.send('<out><dir>' + tmp[0] + '</dir><file>' + tmp[1] + '</file></out>');
            }).catch(function () {
                res.sendStatus(500);
            });
            break;
        case 'c':
            var fname = toolHome('annotationCache', 'DirLists', collection + '.txt');
            if (!fs.existsSync(fname)) {
                res.sendStatus(404);
                return;
            }
            fs.readFile(fname).then(function (val) {
                var tmp = val.toString();
                var images = [];
                var folders = [];
                var temp = tmp.split('\n');
                for (var i = 0; i < temp.length; temp++) {
                    var val = temp[i].split(',');
                    images.push(val[1]);
                    folders.push(val[0]);
                }
                res.send('<out><dir>' + folders[image - 1] + '</dir><file>' + images[image - 1] + '</file></out>');
            }).catch(function () {
                res.sendStatus(500);
            });
            break;
        case 'f':
            var fname = toolHome('Images', folder);
            fs.readdir(fname).then(function (images) {
                var regex = RegExp(image + '$');
                images.filter(function (val) {
                    return val.match(regex);
                });
                res.send('<out><dir>' + folder + '</dir><file>' + images[0] + '</file></out>');
            }).catch(function () {
                res.sendStatus(404);
            });
            break;
    }
});
// TO DO: error handeling
app.post('/annotationTools/perl/submit.cgi', function (req, res) {
    var filename = req.body.annotation.filename[0].split('.');
    filename.length > 0 ? filename.pop() : '';
    filename = filename.join('.');
    var foldername = req.body.annotation.folder[0];
    var globalCount = req.body.annotation.private[0].global_count[0];
    var username = req.body.annotation.private[0].pri_username[0];
    var edited = req.body.annotation.private[0].edited[0];
    var oldName = req.body.annotation.private[0].old_name[0];
    var newName = req.body.annotation.private[0].new_name[0];
    var modifiedControlPoints = req.body.annotation.private[0].modified_cpts[0];
    var videoMode = req.body.annotation.private[0].video[0];
    if (!req.body.annotation.imagesize) {
        req.body.annotation.imagesize = {
            ncols: [require('child_process').spawnSync('identify', ['-format', '"%h"', toolHome('Images', foldername, filename + '.jpg')]).stdout.toString().replace(/\D/g, '')],
            nrows: [require('child_process').spawnSync('identify', ['-format', '"%w"', toolHome('Images', foldername, filename + '.jpg')]).stdout.toString().replace(/\D/g, '')]
        };
    }
    var p = toolHome('Annotations');
    var tmpPath = toolHome('annotationCache', 'TmpAnnotations');
    fs.writeFile(path.join(tmpPath, foldername, filename + '.xml'), toXML(req.body), function (err) {
        fs.copyFile(path.join(tmpPath, foldername, filename + '.xml'), path.join(p, foldername, filename + '.xml'), function (err) {
            res.sendStatus(200);
        });
    });
});

app.post('/annotationTools/perl/write_logfile.cgi', function (req, res) {
    res.setHeader('Content-type', 'text/xml');
    res.send('<nop/>\n');
    res.end();
});

app.get('/Images/:folder/:image', function (req, res) {
    lwip.open(toolHome('Images', req.params.folder, req.params.image), 'jpg', function (err, img) {
        if (err) {
            res.sendStatus(500);
            return;
        }
        var batch = img.batch();
        var index = 0;
        if (fs.existsSync(toolHome('Annotations', req.params.folder, path.parse(req.params.image).name + '.xml')))
            fs.readFile(toolHome('Annotations', req.params.folder, path.parse(req.params.image).name + '.xml'), function (err, anns) {
                anns = parseXML(anns.toString());
                anns = anns.root.children;
                var masks = [];
                anns.forEach(function (val) {
                    if (val.name == 'object')
                        val.children.forEach(function (val) {
                            if (val.name == 'segm')
                                val.children.forEach(function (val) {
                                    if (val.name == 'mask')
                                        masks.push(val.content);
                                })
                        })
                });
                let read = function (indx) {
                    lwip.open(toolHome('Masks', req.params.folder, masks[indx]), function (err, mask) {
                        batch.paste(0, 0, mask);
                        indx != masks.length - 1 ? read(++indx) : (function () {
                            batch = batch.exec(function (err, image) {
                                if (err) {
                                    res.sendStatus(500);
                                    return;
                                }
                                image.toBuffer('jpg', function (err, data) {
                                    res.contentType('image/jpeg');
                                    res.end(data);
                                });
                            });
                        })();
                    });
                }
                read(0);
            });
        else
            res.sendFile(toolHome('Images', req.params.folder, req.params.image));
    });
});
app.use(express.static(path.normalize(path.join(__dirname, '..', '..'))));

app.use(function (req, res) {
    res.sendStatus(404);
});

app.listen(8080);

function toolHome(...paths) {
    return path.normalize(path.join(__dirname, '..', '..', ...paths));
}