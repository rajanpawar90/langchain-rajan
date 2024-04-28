import React from "react";

export interface ColumnContainerProps {
  children: React.ReactNode;
}

export interface ColumnProps {
  children: React.ReactNode;
  className?: string;
}

export const ColumnContainer: React.FC<ColumnContainerProps> = ({ children }) => {
  return (
    <div style={{ display: "flex", flexWrap: "wrap" }}>
      {React.Children.map(children, (child, index) => (
        <div key={index} style={{ flex: "1 0 300px", padding: "10px", overflowX: "clip", zoom: "80%" }}>
          {child}
        </div>
      ))}
    </div>
  );
};

export const Column: React.FC<ColumnProps> = ({ children, className }) => {
  return <div className={className} style={{ flex: "1 0 150px", padding: "10px", overflowX: "clip", zoom: "80%" }}>{children}</div>;
};
