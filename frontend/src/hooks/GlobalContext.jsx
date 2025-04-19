import React, { createContext, useContext, useState } from "react";

const GlobalContext = createContext();

export const GlobalContextProvider = ({ children }) => {
  const [activeTab, setActiveTab] = useState("HOME");
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  const changeTab = (tabName) => {
    setActiveTab(tabName);
  };

  const changeMenu = () => {
    setIsMenuOpen(!isMenuOpen);
  };

  return (
    <GlobalContext.Provider
      value={{ activeTab, changeTab, isMenuOpen, changeMenu }}
    >
      {children}
    </GlobalContext.Provider>
  );
};

export const useGlobalContext = () => useContext(GlobalContext);
