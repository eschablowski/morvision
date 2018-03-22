import React from 'react';
import ReactDOM from 'react-dom';
import {
    BrowserRouter as Router,
    Route,
    Switch
} from 'react-router-dom';
import { Provider } from 'react-redux';

import asyncComponent from 'components/Basic/asyncComponent';
import NavBar from 'components/Basic/NavBar';
import store from 'state';
import 'style/main.scss';

let Index = asyncComponent(() =>
    import('screens/Index')
);
let Submit = asyncComponent(() =>
    import('screens/Submit')
);
let Download = asyncComponent(() =>
    import('screens/Download')
);

class MorVision extends React.Component {
    render() {
        return (
            <Provider store={store}>
                <Router>
                    <div>
                        <header className="header">
                            <NavBar />
                        </header>

                        <section className="content">
                            <Switch>
                                <Route exact path='/' component={Index} />
                                <Route exact path='/Submit' component={Submit} />
                                <Route exact path='/Download' component={Download} />
                            </Switch>
                        </section>
                    </div>
                </Router>
            </Provider>
        )
    }
}
ReactDOM.render(<MorVision />, document.getElementById('root'));