import React from 'react';
import { Link } from 'react-router-dom';
import { connect } from 'react-redux';

class NavBarInternal extends React.Component {
    render() {
        let RightBar = this.state.loggedIn?
        [<li>
            <Link to='/Submit'>
                Submit
            </Link>
        </li>,
        <li>
            <Link to='/Download'>
                Download
            </Link>
        </li>,
        <li>
            <Link to='/About'>
                About
            </Link>
        </li>,
        <li>
            <Link to='/Me'>
                My Account
            </Link>
        </li>,
        <li>
            <a href='//morteam.com/logout?vision'>
                Log Out
            </a>
        </li>]
        :
        [<li>
            <a href='//morteam.com/login?vision'>
                Log In
            </a>
        </li>,
        <li>
            <Link to='/About'>
                About
            </Link>
        </li>];
        return (
            <nav>
                <div className='Left'>
                    <Link to='/'>
                        <h1>
                            MorVision
                        </h1>
                    </Link>
                </div>
                <ul className='Right'>
                    { React.Children.map(RightBar,child=>child) }
                </ul>
            </nav>
        )
    }
    constructor(props) {
        super(props);
        console.log(props);
        this.state = {
            loggedIn: typeof props.user === 'object',
        }
    }
}
function mapStateToProps(state) {
    console.log(state);
    return {
        user: state.init.user
    }    
}
export default connect(mapStateToProps)(NavBarInternal);