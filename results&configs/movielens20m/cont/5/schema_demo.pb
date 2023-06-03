feature {
  name: "session_id"
  type: INT
  int_domain {
    name: "session_id"
    min: 1
    max: 138494 
    is_categorical: false
  }
  annotation {
    tag: "groupby_col"
  }
}
feature {
  name: "item_id-list_seq"
  value_count {
    min: 2
    max: 100
  }
  type: INT
  int_domain {
    name: "item_id-list_seq"
    min: 1
    max: 138493
    is_categorical: true
  }
  annotation {
    tag: "item_id"
    tag: "list"
    tag: "categorical"
    tag: "item"
  }
}
feature {
  name: "rating-list_seq"
  value_count {
    min: 2
    max: 100
  }
  type: FLOAT
  float_domain {
    name: "rating-list_seq"
    min: 0
    max: 5
  }
  annotation {
    tag: "continuous"
    tag: "list"
  }
}
feature {
  name: "genome_relevance-list_seq"
  value_count {
    min: 2
    max: 100
  }
  type: FLOAT
  float_domain {
    name: "genome_relevance-list_seq"
    min: 0
    max: 1
  }
  annotation {
    tag: "continuous"
    tag: "list"
  }
}
feature {
  name: "_genres-list_seq"
  value_count {
    min: 2
    max: 100
  }
  type: FLOAT
  float_domain {
    name: "_genres-list_seq"
    min: 0
    max: 21
  }
  annotation {
    tag: "categorical"
    tag: "list"
  }
}
feature {
  name: "genome_tag-list_seq"
  value_count {
    min: 2
    max: 100
  }
  type: FLOAT
  float_domain {
    name: "genome_tag-list_seq"
    min: 0
    max: 815
  }
  annotation {
    tag: "categorical"
    tag: "list"
  }
}
feature {
  name: "tag-list_seq"
  value_count {
    min: 2
    max: 100
  }
  type: FLOAT
  float_domain {
    name: "tag-list_seq"
    min: 0
    max: 17942
  }
  annotation {
    tag: "categorical"
    tag: "list"
  }
}
feature {
  name: "et_dayofday-list_seq"
  value_count {
    min: 2
    max: 100
  }
  type: FLOAT
  float_domain {
    name: "et_dayofday-list_seq"
    min: 1
    max: 31
  }
  annotation {
    tag: "continuous"
    tag: "list"
  }
}
feature {
  name: "et_dayofweek-list_seq"
  value_count {
    min: 2
    max: 100
  }
  type: FLOAT
  float_domain {
    name: "et_dayofweek-list_seq"
    min: 0
    max: 6
  }
  annotation {
    tag: "continuous"
    tag: "list"
  }
}
feature {
  name: "et_year-list_seq"
  value_count {
    min: 2
    max: 100
  }
  type: FLOAT
  float_domain {
    name: "et_year-list_seq"
    min: 1995
    max: 2015
  }
  annotation {
    tag: "continuous"
    tag: "list"
  }
}
feature {
  name: "et_dayofweek_sin-list_seq"
  value_count {
    min: 2
    max: 100
  }
  type: FLOAT
  float_domain {
    name: "et_dayofweek_sin-list_seq"
    min: -1
    max: 1
  }
  annotation {
    tag: "continuous"
    tag: "time"
    tag: "list"
  }
}
