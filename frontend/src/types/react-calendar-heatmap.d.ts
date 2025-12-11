declare module 'react-calendar-heatmap' {
  import * as React from 'react';

  export interface CalendarHeatmapValue {
    date: string | Date;
    count?: number;
    [key: string]: any;
  }

  export interface CalendarHeatmapProps {
    startDate: Date | string;
    endDate: Date | string;
    values: CalendarHeatmapValue[];
    classForValue?: (value: CalendarHeatmapValue) => string;
    titleForValue?: (value: CalendarHeatmapValue) => string | undefined;
    tooltipDataAttrs?: any;
    gutterSize?: number;
    showWeekdayLabels?: boolean;
    transformDayElement?: (element: React.ReactElement, value: CalendarHeatmapValue) => React.ReactElement;
  }

  export default class CalendarHeatmap extends React.Component<CalendarHeatmapProps> {}
}
