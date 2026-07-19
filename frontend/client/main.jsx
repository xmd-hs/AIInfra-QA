import React, { Component } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }
  static getDerivedStateFromError(error) { return { error }; }
  componentDidCatch(error, info) { console.error("React rendering error", error, info); }
  render() {
    if (this.state.error) {
      return <div className="fatal-error"><div><span>!</span><h1>页面遇到了兼容性问题</h1><p>{this.state.error.message}</p><button onClick={() => location.reload()}>重新加载页面</button></div></div>;
    }
    return this.props.children;
  }
}

createRoot(document.getElementById("root")).render(<ErrorBoundary><App /></ErrorBoundary>);
