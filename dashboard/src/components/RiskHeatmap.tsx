/**
 * 🔥 RISK HEATMAP
 * Interactive risk visualization
 */
import React, { useEffect, useRef } from 'react';
import * as d3 from 'd3';

interface HeatmapData {
  customer_id: string;
  date: string;
  risk_score: number;
  transaction_count: number;
}

export const RiskHeatmap: React.FC<{ data: HeatmapData[] }> = ({ data }) => {
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!svgRef.current || !data.length) return;

    // Clear
    d3.select(svgRef.current).selectAll('*').remove();

    // Dimensions
    const margin = { top: 50, right: 50, bottom: 100, left: 100 };
    const width = 1000 - margin.left - margin.right;
    const height = 600 - margin.top - margin.bottom;

    const svg = d3.select(svgRef.current)
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Get unique customers and dates
    const customers = Array.from(new Set(data.map(d => d.customer_id)));
    const dates = Array.from(new Set(data.map(d => d.date))).sort();

    // Scales
    const x = d3.scaleBand()
      .range([0, width])
      .domain(dates)
      .padding(0.05);

    const y = d3.scaleBand()
      .range([height, 0])
      .domain(customers)
      .padding(0.05);

    const colorScale = d3.scaleSequential()
      .interpolator(d3.interpolateRdYlGn)
      .domain([1, 0]); // Reversed: high risk = red

    // Add X axis
    svg.append('g')
      .attr('transform', `translate(0,${height})`)
      .call(d3.axisBottom(x))
      .selectAll('text')
      .attr('transform', 'rotate(-45)')
      .style('text-anchor', 'end');

    // Add Y axis
    svg.append('g')
      .call(d3.axisLeft(y));

    // Tooltip
    const tooltip = d3.select('body')
      .append('div')
      .style('position', 'absolute')
      .style('visibility', 'hidden')
      .style('background', 'white')
      .style('border', '1px solid #ddd')
      .style('border-radius', '4px')
      .style('padding', '10px')
      .style('box-shadow', '0 2px 4px rgba(0,0,0,0.1)');

    // Create cells
    svg.selectAll()
      .data(data)
      .join('rect')
      .attr('x', d => x(d.date) || 0)
      .attr('y', d => y(d.customer_id) || 0)
      .attr('width', x.bandwidth())
      .attr('height', y.bandwidth())
      .style('fill', d => colorScale(d.risk_score))
      .style('stroke', 'white')
      .style('stroke-width', 2)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this)
          .style('stroke', 'black')
          .style('stroke-width', 3);

        tooltip
          .style('visibility', 'visible')
          .html(`
            <strong>${d.customer_id}</strong><br/>
            Date: ${d.date}<br/>
            Risk Score: ${(d.risk_score * 100).toFixed(1)}%<br/>
            Transactions: ${d.transaction_count}
          `);
      })
      .on('mousemove', function(event) {
        tooltip
          .style('top', (event.pageY - 10) + 'px')
          .style('left', (event.pageX + 10) + 'px');
      })
      .on('mouseout', function() {
        d3.select(this)
          .style('stroke', 'white')
          .style('stroke-width', 2);

        tooltip.style('visibility', 'hidden');
      });

    // Add title
    svg.append('text')
      .attr('x', width / 2)
      .attr('y', -20)
      .attr('text-anchor', 'middle')
      .style('font-size', '18px')
      .style('font-weight', 'bold')
      .text('Customer Risk Heatmap Over Time');

    // Add legend
    const legendWidth = 300;
    const legendHeight = 20;

    const legend = svg.append('g')
      .attr('transform', `translate(${width - legendWidth}, ${height + 60})`);

    const legendScale = d3.scaleLinear()
      .domain([0, 1])
      .range([0, legendWidth]);

    const legendAxis = d3.axisBottom(legendScale)
      .ticks(5)
      .tickFormat(d => `${(d * 100).toFixed(0)}%`);

    // Create gradient
    const defs = svg.append('defs');
    const linearGradient = defs.append('linearGradient')
      .attr('id', 'legend-gradient');

    linearGradient.selectAll('stop')
      .data([
        { offset: '0%', color: colorScale(1) },
        { offset: '50%', color: colorScale(0.5) },
        { offset: '100%', color: colorScale(0) }
      ])
      .join('stop')
      .attr('offset', d => d.offset)
      .attr('stop-color', d => d.color);

    legend.append('rect')
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#legend-gradient)');

    legend.append('g')
      .attr('transform', `translate(0,${legendHeight})`)
      .call(legendAxis);

    legend.append('text')
      .attr('x', -10)
      .attr('y', legendHeight / 2)
      .attr('dy', '0.35em')
      .attr('text-anchor', 'end')
      .text('Low Risk');

    legend.append('text')
      .attr('x', legendWidth + 10)
      .attr('y', legendHeight / 2)
      .attr('dy', '0.35em')
      .text('High Risk');

    // Cleanup
    return () => {
      tooltip.remove();
    };
  }, [data]);

  return (
    <div className="risk-heatmap p-4 bg-white rounded-lg shadow">
      <svg ref={svgRef}></svg>
    </div>
  );
};

