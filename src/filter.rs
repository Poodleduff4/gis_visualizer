use std::fmt;

use crate::gis_layer::AttributeValue;

#[derive(Default, PartialEq, Clone, Copy)]
pub enum FilterLogic {
    #[default]
    And,
    Or,
}

#[derive(PartialEq, Clone)]
pub enum FilterOperation {
    LessThan,
    GreaterThan,
    Equal,
}

impl fmt::Display for FilterOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterOperation::Equal => write!(f, "="),
            FilterOperation::GreaterThan => write!(f, ">"),
            FilterOperation::LessThan => write!(f, "<"),
        }
    }
}

pub struct LayerAttributeFilter {
    pub attribute: Option<String>,
    pub operation: Option<FilterOperation>,
    pub comparitor: AttributeValue,
    pub comparitor_raw: String,
}
