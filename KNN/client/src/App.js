import React, { Component } from 'react';
import axios from 'axios'
import logo from './logo.svg';
import './App.css';

class App extends Component {

  state  = {
          a: '',
          b: '',
          c: ''
            }

  textInput =  () => {

      this.setState({a: ""});
      this.setState({b: ""});
      this.setState({c: ""});

      const reviewText1 = document.getElementById('review-text1');
      const text1 = reviewText1.value;
      console.log("TEXT1: ", text1);

      this.setState({a: text1});

      const reviewText2 = document.getElementById('review-text2');
      const text2 =  reviewText2.value;
      console.log("TEXT2: ", text2);

      this.setState({b: text2});

      const str3 = "/api/user/" + `${[text1]}`+"-"+`${[text2]}`;
      console.log("xxxxx: ", `${[text1]}`+"-"+`${[text2]}`);

      axios.get(str3)
      .then( response =>{

      let z = response.data;

      console.log("RE: ", z);
      this.setState({c: z});

    })

  };  


  render() {
    return (

    <div > 
      <div className="App">
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
           <h3> This app is using KNN algortithim with tfjs-node to predict adult person gender based on weight & height</h3>
        </header>
        <p className="App-intro">
            Height: 
            {this.state.a}
            <br/>
            Weight: 
            {this.state.b}
            <br/>
            Gender: 
            {this.state.c}
        </p>
      </div>

      <div id = 'textarea'>
        <form>
              <textarea id = 'review-text1' name="input" cols="15" rows="2" placeholder="Height (cm)"></textarea>
         </form>
         <br/>   
         <form>
              <textarea  id = 'review-text2' name="input" cols="15" rows="2" placeholder="Weight (kg)"></textarea>
         </form>
              <p id = "submit" type="submit"  onClick = {this.textInput} >Submit</p>
      </div>   
    </div> 
          
    );
  }
}

export default App;
