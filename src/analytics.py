import numpy as np
import rasterio
import math

def calculate_areas(class_counts, transform, crs):
    """
    Calculates land-use area in Hectares and Acres based on pixel counts.
    Assumes transform is in meters (standard for Orthos).
    """
    # GSD (Ground Sampling Distance)
    res_x_raw = abs(transform[0])
    res_y_raw = abs(transform[4])
    
    # Handle EPSG:4326 degree to meters
    if crs and '4326' in str(crs):
        lat = transform[5]
        res_x_m = res_x_raw * 111320.0 * math.cos(math.radians(lat))
        res_y_m = res_y_raw * 111320.0
        pixel_area = res_x_m * res_y_m
    else:
        pixel_area = res_x_raw * res_y_raw
        
    
    # Area conversion factors
    sqm_to_ha = 10000.0
    sqm_to_acre = 4046.856
    
    areas = {}
    for i, count in enumerate(class_counts):
        sq_meters = float(count) * pixel_area
        areas[i] = {
            "sq_meters": sq_meters,
            "hectares": sq_meters / sqm_to_ha,
            "acres": sq_meters / sqm_to_acre
        }
    
    return areas

def get_municipal_insights(areas):
    """
    Generates actionable insights for urban planners.
    0: Background, 1: Road, 2: Built-up, 3: Water
    """
    # Areas might come from JSON (string keys) or Dict (int keys)
    areas_processed = {str(k): v for k, v in areas.items()}
    
    total_area_ha = sum([a["hectares"] for a in areas_processed.values()])
    if total_area_ha == 0:
        return "N/A"
        
    urban_density = (areas_processed["2"]["hectares"] / total_area_ha) * 100
    road_connectivity = (areas_processed["1"]["hectares"] / total_area_ha) * 100
    green_space_ratio = (areas_processed["0"]["hectares"] / total_area_ha) * 100
    water_security = (areas_processed["3"]["hectares"] / total_area_ha) * 100
    
    insights = {
        "urban_density_score": urban_density,
        "road_connectivity_score": road_connectivity,
        "green_space_ratio": green_space_ratio,
        "water_security_ratio": water_security,
        "assessment": "",
        "recommendations": []
    }
    
    # Urban Sprawl Assessment
    if urban_density > 40:
        insights["assessment"] = "High Urbanization"
        insights["recommendations"].extend([
            "Prioritize vertical infrastructure development to mitigate horizontal sprawl.",
            "Implement high-density residential zoning in core municipal zones.",
            "Accelerate public mass-transit corridors to support high population concentration."
        ])
    elif urban_density < 10:
        insights["assessment"] = "Low Urbanization"
        insights["recommendations"].extend([
            "Significant headroom for sustainable greenfield development.",
            "Establish strict environmental preservation buffers before expansion.",
            "Focus on decentralized essential service hubs to prevent future congestion."
        ])
    else:
        insights["assessment"] = "Moderate Urbanization"
        insights["recommendations"].extend([
            "Implement mixed-use zoning to ensure balanced residential and commercial growth.",
            "Incorporate Transit-Oriented Development (TOD) frameworks for upcoming infrastructure.",
            "Establish dedicated transition zones between developed and agrarian land-use.",
            "Strategy: Balanced growth with a focus on core-peripheral connectivity."
        ])
        
    # Infrastructure Needs
    if road_connectivity < 2.5:
        insights["recommendations"].append("Sub-optimal transport density detected. Expand primary and secondary arterial network.")
    
    # Environmental Conservation
    if green_space_ratio < 25:
        insights["recommendations"].append("Ecological deficit identified. Mandate 20% minimum green-cover per new development plot.")
    
    if water_security > 5:
         insights["recommendations"].append("Substantial hydrographic assets. Optimize for sustainable water-front planning and runoff management.")
        
    return insights

def format_class_data_for_plotly(areas, class_names):
    """Formats area data for bar/pie charts."""
    areas_processed = {str(k): v for k, v in areas.items()}
    names = [class_names[i] for i in range(len(class_names))]
    hectares = [areas_processed[str(i)]["hectares"] for i in range(len(class_names))]
    
    return names, hectares
