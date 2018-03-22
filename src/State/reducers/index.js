import { combineReducers } from 'redux';

export default combineReducers({
    init: state=>state||{user:{name:'bob'}}
});