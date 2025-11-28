/**
 * 🕸️ NETWORK VISUALIZATION
 * D3.js interactive network graph
 */
import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface NetworkNode {
  id: string;
  label: string;
  risk_score: number;
  transaction_count: number;
  is_suspicious: boolean;
}

interface NetworkEdge {
  source: string;
  target: string;
  amount: number;
  timestamp: string;
}

interface NetworkData {
  nodes: NetworkNode[];
  edges: NetworkEdge[];
}

export const NetworkVisualization: React.FC<{ data: NetworkData }> = ({ data }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedNode, setSelectedNode] = useState<NetworkNode | null>(null);

  useEffect(() => {
    if (!svgRef.current || !data.nodes.length) return;

    // Clear existing
    d3.select(svgRef.current).selectAll('*').remove();

    // Dimensions
    const width = 1200;
    const height = 800;

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height);

    // Create force simulation
    const simulation = d3.forceSimulation(data.nodes as any)
      .force('link', d3.forceLink(data.edges)
        .id((d: any) => d.id)
        .distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius(30));

    // Create arrow marker
    svg.append('defs').selectAll('marker')
      .data(['arrow'])
      .join('marker')
      .attr('id', 'arrow')
      .attr('viewBox', '0 -5 10 10')
      .attr('refX', 20)
      .attr('refY', 0)
      .attr('markerWidth', 6)
      .attr('markerHeight', 6)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-5L10,0L0,5')
      .attr('fill', '#999');

    // Create edges
    const link = svg.append('g')
      .selectAll('line')
      .data(data.edges)
      .join('line')
      .attr('stroke', '#999')
      .attr('stroke-opacity', 0.6)
      .attr('stroke-width', (d: any) => Math.sqrt(d.amount / 1000))
      .attr('marker-end', 'url(#arrow)');

    // Create nodes
    const node = svg.append('g')
      .selectAll('circle')
      .data(data.nodes)
      .join('circle')
      .attr('r', (d: any) => 10 + (d.transaction_count / 10))
      .attr('fill', (d: any) => 
        d.is_suspicious ? '#ef4444' : 
        d.risk_score > 0.7 ? '#f59e0b' : 
        '#10b981'
      )
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .call(d3.drag<any, any>()
        .on('start', dragStarted)
        .on('drag', dragged)
        .on('end', dragEnded))
      .on('click', (event, d: any) => {
        setSelectedNode(d);
      });

    // Add labels
    const label = svg.append('g')
      .selectAll('text')
      .data(data.nodes)
      .join('text')
      .text((d: any) => d.label)
      .attr('font-size', 10)
      .attr('dx', 15)
      .attr('dy', 4);

    // Update positions
    simulation.on('tick', () => {
      link
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      node
        .attr('cx', (d: any) => d.x)
        .attr('cy', (d: any) => d.y);

      label
        .attr('x', (d: any) => d.x)
        .attr('y', (d: any) => d.y);
    });

    // Drag functions
    function dragStarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragEnded(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }

    // Cleanup
    return () => {
      simulation.stop();
    };
  }, [data]);

  return (
    <div className="network-visualization">
      <div className="controls mb-4 flex gap-4">
        <button className="btn btn-primary">
          🔍 Zoom In
        </button>
        <button className="btn btn-secondary">
          🔍 Zoom Out
        </button>
        <button className="btn btn-secondary">
          🔄 Reset
        </button>
        <button className="btn btn-secondary">
          💾 Export
        </button>
      </div>

      <div className="graph-container border rounded-lg shadow-lg bg-white">
        <svg ref={svgRef} className="w-full"></svg>
      </div>

      {selectedNode && (
        <div className="node-details mt-4 p-4 border rounded-lg bg-white shadow">
          <h3 className="text-lg font-bold mb-2">Node Details</h3>
          <div className="grid grid-cols-2 gap-2">
            <div>ID:</div>
            <div className="font-mono">{selectedNode.id}</div>
            <div>Label:</div>
            <div>{selectedNode.label}</div>
            <div>Risk Score:</div>
            <div className="font-bold">
              <span className={
                selectedNode.risk_score > 0.7 ? 'text-red-600' :
                selectedNode.risk_score > 0.4 ? 'text-yellow-600' :
                'text-green-600'
              }>
                {(selectedNode.risk_score * 100).toFixed(1)}%
              </span>
            </div>
            <div>Transactions:</div>
            <div>{selectedNode.transaction_count}</div>
            <div>Status:</div>
            <div>
              {selectedNode.is_suspicious ? (
                <span className="text-red-600 font-bold">⚠️ SUSPICIOUS</span>
              ) : (
                <span className="text-green-600">✅ Clear</span>
              )}
            </div>
          </div>
        </div>
      )}

      <div className="legend mt-4 p-4 border rounded-lg bg-gray-50">
        <h4 className="font-bold mb-2">Legend</h4>
        <div className="flex gap-6">
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-red-500"></div>
            <span>Suspicious</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-yellow-500"></div>
            <span>High Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 rounded-full bg-green-500"></div>
            <span>Low Risk</span>
          </div>
        </div>
      </div>
    </div>
  );
};

