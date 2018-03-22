import React from 'react';
export default class Submit extends React.Component {
    render() {
        return (
            <form onSubmit={this.handleSubmit.bind(this)}>
                <input type="file"
                    ref={input => {
                        this.fileInput = input;
                    }} />
                <input type="submit" />
                <Error data={this.state.error} show={this.state.submitted} />
            </form>
        );
    }
    handleSubmit(event) {
        event.preventDefault();
        if (this.fileInput.files.length === 0)
            return;
        var fd = new FormData();
        fd.append('image', this.fileInput.files[0].slice(), this.fileInput.files[0].name);

        fetch('/Submit?'+this.fileInput.files[0].name,
            {
                method: 'post',
                body: fd
            })
            .then((function (oEvent) {
                this.setState(Object.assign({}, this.state, {
                    submitted: true,
                    error: false,
                }));
            }).bind(this))
            .catch(function (err) {
                console.log(err);
            });
    }
    constructor(props) {
        super(props);
        this.state = {
            error: false,
            submitted: false,
        };
    }
}
export class Error extends React.Component {
    render() {
        return this.props.show ? null : (
            <h2>
                {this.props.data ? 'Error' : 'Submitted'}
            </h2>
        );
    }
}