import React from 'react';
import { connect } from 'react-redux';
class Download extends React.Component {
    render() {
        return (
            <div>
                <button>
                    <a href="/Download?all" target='_blank'>
                        All
                    </a>
                </button>
                <button>
                    <a href={"/Download?" + this.props.name} target='_blank'>
                        Self
                    </a>
                </button>
            </div>
        );
    }
};

function mapStateToProps(state) {
    return {
        name: state.init.user.name,
    };
}

export default connect(mapStateToProps)(Download);